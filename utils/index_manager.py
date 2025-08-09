"""
IndexManager for BYEOLI_TALK_AT_GNH_app.

Singleton class that manages all FAISS vector stores with hot-swapping capabilities.
Preloads all indices at app startup and monitors file changes for automatic reloading.
"""

import os
import time
import hashlib
import threading
import pickle
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import faiss
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from utils.config import get_config
from utils.contracts import ComponentStatus, HealthCheck


logger = logging.getLogger(__name__)


@dataclass
class IndexInfo:
    """Information about a loaded index."""
    name: str
    faiss_index: faiss.Index
    metadata: Dict[str, Any]
    file_hash: str
    loaded_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class VectorStoreFileHandler(FileSystemEventHandler):
    """File system event handler for vectorstore directory monitoring."""
    
    def __init__(self, index_manager: 'IndexManager'):
        self.index_manager = index_manager
        self.debounce_time = 2.0  # seconds
        self.pending_reloads = {}
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process .faiss and .pkl files
        if file_path.suffix not in ['.faiss', '.pkl']:
            return
        
        # Extract index name from path
        index_name = self._extract_index_name(file_path)
        if not index_name:
            return
        
        # Debounce rapid file changes
        current_time = time.time()
        if index_name in self.pending_reloads:
            if current_time - self.pending_reloads[index_name] < self.debounce_time:
                return
        
        self.pending_reloads[index_name] = current_time
        
        # Schedule reload after debounce period
        threading.Timer(
            self.debounce_time,
            self._trigger_reload,
            args=[index_name]
        ).start()
    
    def _extract_index_name(self, file_path: Path) -> Optional[str]:
        """Extract index name from file path."""
        try:
            # Expected path: vectorstores/vectorstore_<name>/...
            parts = file_path.parts
            for i, part in enumerate(parts):
                if part.startswith('vectorstore_'):
                    return part.replace('vectorstore_', '')
            return None
        except Exception:
            return None
    
    def _trigger_reload(self, index_name: str):
        """Trigger index reload if file actually changed."""
        try:
            # Check if index is still pending reload (not already processed)
            if index_name in self.pending_reloads:
                del self.pending_reloads[index_name]
                self.index_manager._reload_index_if_changed(index_name)
        except Exception as e:
            logger.error(f"Failed to reload index {index_name}: {e}")


class IndexManager:
    """Singleton manager for all FAISS vector indices."""
    
    _instance: Optional['IndexManager'] = None
    _initialized: bool = False
    _lock = threading.RLock()
    
    def __new__(cls) -> 'IndexManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self):
        with self._lock:
            if not self._initialized:
                self.config = get_config()
                self.vectorstores_dir = self.config.paths.get_absolute_path(
                    self.config.paths.vectorstores_dir
                )
                
                # Index storage
                self.indices: Dict[str, IndexInfo] = {}
                self.loading_status: Dict[str, bool] = {}
                self.load_errors: Dict[str, str] = {}
                
                # File monitoring
                self.observer: Optional[Observer] = None
                self.file_handler: Optional[VectorStoreFileHandler] = None
                
                # Thread safety
                self.access_lock = threading.RLock()
                
                IndexManager._initialized = True
                logger.info("IndexManager initialized")
    
    def preload_all_indices(self, max_workers: int = 3) -> Dict[str, bool]:
        """
        Preload all available indices in parallel.
        
        Args:
            max_workers: Maximum number of threads for parallel loading
            
        Returns:
            Dict mapping index names to success status
        """
        logger.info("Starting preload of all indices...")
        start_time = time.time()
        
        # Discover available indices
        available_indices = self._discover_indices()
        if not available_indices:
            logger.warning("No indices found in vectorstores directory")
            return {}
        
        logger.info(f"Found {len(available_indices)} indices to load: {list(available_indices.keys())}")
        
        # Load indices in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit loading tasks
            future_to_index = {
                executor.submit(self._load_single_index, name, path): name
                for name, path in available_indices.items()
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index_name = future_to_index[future]
                try:
                    success = future.result()
                    results[index_name] = success
                    if success:
                        logger.info(f"Successfully loaded index: {index_name}")
                    else:
                        logger.error(f"Failed to load index: {index_name}")
                except Exception as e:
                    logger.error(f"Exception loading index {index_name}: {e}")
                    results[index_name] = False
        
        # Start file monitoring
        self._start_file_monitoring()
        
        elapsed_time = time.time() - start_time
        successful_loads = sum(1 for success in results.values() if success)
        
        logger.info(
            f"Preload completed in {elapsed_time:.2f}s. "
            f"Loaded {successful_loads}/{len(available_indices)} indices successfully."
        )
        
        return results
    
    def get_index(self, name: str) -> Optional[Tuple[faiss.Index, Dict[str, Any]]]:
        """
        Get FAISS index and metadata by name.
        
        Args:
            name: Index name (e.g., 'general', 'publish', 'satisfaction')
            
        Returns:
            Tuple of (faiss_index, metadata) or None if not found
        """
        with self.access_lock:
            if name not in self.indices:
                # Try to load on-demand
                if not self._load_index_on_demand(name):
                    logger.warning(f"Index '{name}' not available")
                    return None
            
            index_info = self.indices[name]
            index_info.update_access()
            
            return index_info.faiss_index, index_info.metadata
    
    def get_available_indices(self) -> List[str]:
        """Get list of currently loaded index names."""
        with self.access_lock:
            return list(self.indices.keys())
    
    def get_index_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all loaded indices."""
        with self.access_lock:
            stats = {}
            for name, info in self.indices.items():
                stats[name] = {
                    'loaded_at': info.loaded_at,
                    'access_count': info.access_count,
                    'last_accessed': info.last_accessed,
                    'file_hash': info.file_hash[:8],  # Short hash
                    'index_size': info.faiss_index.ntotal if info.faiss_index else 0,
                    'metadata': {k: v for k, v in info.metadata.items() 
                               if k not in ['embeddings', 'texts']}  # Exclude large data
                }
            return stats
    
    def reload_index(self, name: str, force: bool = False) -> bool:
        """
        Reload a specific index.
        
        Args:
            name: Index name to reload
            force: Force reload even if file hash hasn't changed
            
        Returns:
            True if reload was successful
        """
        with self.access_lock:
            logger.info(f"Reloading index: {name} (force={force})")
            
            if not force and name in self.indices:
                # Check if file actually changed
                current_hash = self._calculate_index_file_hash(name)
                if current_hash and current_hash == self.indices[name].file_hash:
                    logger.info(f"Index {name} file unchanged, skipping reload")
                    return True
            
            # Remove old index if exists
            if name in self.indices:
                del self.indices[name]
            
            # Load fresh index
            available_indices = self._discover_indices()
            if name in available_indices:
                return self._load_single_index(name, available_indices[name])
            else:
                logger.error(f"Index {name} not found in vectorstores directory")
                return False
    
    def get_health_status(self) -> HealthCheck:
        """Get health status of IndexManager."""
        with self.access_lock:
            try:
                # Check overall status
                total_indices = len(self._discover_indices())
                loaded_indices = len(self.indices)
                error_count = len(self.load_errors)
                
                if loaded_indices == 0:
                    overall_status = ComponentStatus.UNHEALTHY
                elif error_count > 0 or loaded_indices < total_indices:
                    overall_status = ComponentStatus.DEGRADED
                else:
                    overall_status = ComponentStatus.HEALTHY
                
                # Component details
                components = {}
                for name in self._discover_indices():
                    if name in self.indices:
                        components[f"index_{name}"] = ComponentStatus.HEALTHY
                    elif name in self.load_errors:
                        components[f"index_{name}"] = ComponentStatus.UNHEALTHY
                    else:
                        components[f"index_{name}"] = ComponentStatus.UNKNOWN
                
                # File monitoring status
                components["file_monitoring"] = (
                    ComponentStatus.HEALTHY if self.observer and self.observer.is_alive()
                    else ComponentStatus.DEGRADED
                )
                
                return HealthCheck(
                    overall_status=overall_status,
                    components=components,
                    details={
                        "total_indices": total_indices,
                        "loaded_indices": loaded_indices,
                        "error_count": error_count,
                        "load_errors": dict(self.load_errors),
                        "monitoring_active": self.observer is not None and self.observer.is_alive()
                    }
                )
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return HealthCheck(
                    overall_status=ComponentStatus.UNKNOWN,
                    components={},
                    details={"error": str(e)}
                )
    
    def shutdown(self):
        """Clean shutdown of IndexManager."""
        logger.info("Shutting down IndexManager...")
        
        # Stop file monitoring
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.observer = None
        
        # Clear indices
        with self.access_lock:
            self.indices.clear()
            self.loading_status.clear()
            self.load_errors.clear()
        
        logger.info("IndexManager shutdown complete")
    
    def _discover_indices(self) -> Dict[str, Path]:
        """Discover available vectorstore directories."""
        available_indices = {}
        
        if not self.vectorstores_dir.exists():
            logger.warning(f"Vectorstores directory not found: {self.vectorstores_dir}")
            return available_indices
        
        for item in self.vectorstores_dir.iterdir():
            if item.is_dir() and item.name.startswith('vectorstore_'):
                index_name = item.name.replace('vectorstore_', '')
                
                # Check for required files
                faiss_file = item / f"{index_name}_index.faiss"
                pkl_file = item / f"{index_name}_index.pkl"
                
                if faiss_file.exists() and pkl_file.exists():
                    available_indices[index_name] = item
                else:
                    logger.warning(
                        f"Index {index_name} missing required files: "
                        f"faiss_exists={faiss_file.exists()}, pkl_exists={pkl_file.exists()}"
                    )
        
        return available_indices
    
    def _load_single_index(self, name: str, index_dir: Path) -> bool:
        """Load a single index from directory."""
        try:
            logger.debug(f"Loading index: {name} from {index_dir}")
            
            # Set loading status
            self.loading_status[name] = True
            
            # File paths
            faiss_file = index_dir / f"{name}_index.faiss"
            pkl_file = index_dir / f"{name}_index.pkl"
            
            # Calculate file hash for change detection
            file_hash = self._calculate_file_hash(faiss_file, pkl_file)
            
            # Load FAISS index
            if not faiss_file.exists():
                raise FileNotFoundError(f"FAISS file not found: {faiss_file}")
            
            faiss_index = faiss.read_index(str(faiss_file))
            logger.debug(f"Loaded FAISS index with {faiss_index.ntotal} vectors")
            
            # Load metadata
            if not pkl_file.exists():
                raise FileNotFoundError(f"Metadata file not found: {pkl_file}")
            
            with open(pkl_file, 'rb') as f:
                metadata = pickle.load(f)
            
            logger.debug(f"Loaded metadata with {len(metadata.get('texts', []))} texts")
            
            # Store index info
            with self.access_lock:
                self.indices[name] = IndexInfo(
                    name=name,
                    faiss_index=faiss_index,
                    metadata=metadata,
                    file_hash=file_hash,
                    loaded_at=time.time()
                )
                
                # Clear any previous errors
                if name in self.load_errors:
                    del self.load_errors[name]
            
            logger.info(f"Successfully loaded index: {name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load index {name}: {str(e)}"
            logger.error(error_msg)
            
            with self.access_lock:
                self.load_errors[name] = error_msg
                if name in self.indices:
                    del self.indices[name]
            
            return False
        finally:
            self.loading_status[name] = False
    
    def _load_index_on_demand(self, name: str) -> bool:
        """Load an index on-demand if not already loaded."""
        if name in self.loading_status and self.loading_status[name]:
            # Already loading, wait a bit
            time.sleep(0.1)
            return name in self.indices
        
        available_indices = self._discover_indices()
        if name in available_indices:
            return self._load_single_index(name, available_indices[name])
        else:
            logger.error(f"Index {name} not found for on-demand loading")
            return False
    
    def _calculate_file_hash(self, *files: Path) -> str:
        """Calculate combined hash of multiple files."""
        hasher = hashlib.md5()
        
        for file_path in files:
            if file_path.exists():
                # Include file modification time and size
                stat = file_path.stat()
                hasher.update(f"{file_path}:{stat.st_mtime}:{stat.st_size}".encode())
        
        return hasher.hexdigest()
    
    def _calculate_index_file_hash(self, name: str) -> Optional[str]:
        """Calculate hash for specific index files."""
        available_indices = self._discover_indices()
        if name not in available_indices:
            return None
        
        index_dir = available_indices[name]
        faiss_file = index_dir / f"{name}_index.faiss"
        pkl_file = index_dir / f"{name}_index.pkl"
        
        return self._calculate_file_hash(faiss_file, pkl_file)
    
    def _start_file_monitoring(self):
        """Start file system monitoring for hot-swap."""
        try:
            if not self.vectorstores_dir.exists():
                logger.warning("Cannot start file monitoring: vectorstores directory not found")
                return
            
            self.file_handler = VectorStoreFileHandler(self)
            self.observer = Observer()
            self.observer.schedule(
                self.file_handler,
                str(self.vectorstores_dir),
                recursive=True
            )
            self.observer.start()
            
            logger.info("File monitoring started for hot-swap capability")
            
        except Exception as e:
            logger.error(f"Failed to start file monitoring: {e}")
    
    def _reload_index_if_changed(self, name: str):
        """Reload index if file has changed (called by file monitor)."""
        try:
            current_hash = self._calculate_index_file_hash(name)
            if not current_hash:
                return
            
            with self.access_lock:
                if name in self.indices and current_hash != self.indices[name].file_hash:
                    logger.info(f"File change detected for index {name}, reloading...")
                    self.reload_index(name, force=True)
                elif name not in self.indices:
                    logger.info(f"New index detected: {name}, loading...")
                    self._load_index_on_demand(name)
        
        except Exception as e:
            logger.error(f"Failed to check/reload index {name}: {e}")


# Global instance
_index_manager: Optional[IndexManager] = None


def get_index_manager() -> IndexManager:
    """Get the global IndexManager instance."""
    global _index_manager
    if _index_manager is None:
        _index_manager = IndexManager()
    return _index_manager


def initialize_index_manager(max_workers: int = 3) -> Dict[str, bool]:
    """Initialize and preload IndexManager."""
    manager = get_index_manager()
    return manager.preload_all_indices(max_workers=max_workers)


def shutdown_index_manager():
    """Shutdown the global IndexManager."""
    global _index_manager
    if _index_manager:
        _index_manager.shutdown()
        _index_manager = None
