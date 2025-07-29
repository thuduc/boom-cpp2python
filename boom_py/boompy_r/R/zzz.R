#' @importFrom reticulate import py_module_available py_install configure_environment use_python
.onLoad <- function(libname, pkgname) {
  # Configure Python
  reticulate::configure_environment(pkgname)
  
  # Check if boom module is available
  if (!reticulate::py_module_available("boom")) {
    # Try to find boom in the package's inst/python directory
    boom_path <- system.file("python", package = "boompy")
    if (boom_path != "") {
      # Add to Python path
      reticulate::py_run_string(sprintf("
import sys
if '%s' not in sys.path:
    sys.path.insert(0, '%s')
", boom_path, boom_path))
    }
  }
}

# Package environment to store Python modules
boom_env <- new.env(parent = emptyenv())

#' Setup BOOM Python Backend
#' 
#' Initialize the BOOM Python modules and check installation.
#' 
#' @param python Path to Python executable (optional)
#' @param required_version Minimum Python version required
#' 
#' @return Invisible NULL. Called for side effects.
#' @export
#' 
#' @examples
#' \dontrun{
#' boom_setup()
#' }
boom_setup <- function(python = NULL, required_version = "3.7") {
  # Use specific Python if provided
  if (!is.null(python)) {
    reticulate::use_python(python, required = TRUE)
  }
  
  # Check Python version
  py_config <- reticulate::py_config()
  if (is.null(py_config$version)) {
    stop("Python not found. Please install Python >= ", required_version)
  }
  
  # Import Python modules
  tryCatch({
    boom_env$boom <- reticulate::import("boom", delay_load = FALSE)
    boom_env$np <- reticulate::import("numpy", delay_load = FALSE)
    boom_env$Vector <- boom_env$boom$linalg$Vector
    boom_env$Matrix <- boom_env$boom$linalg$Matrix
    
    message("BOOM Python backend loaded successfully")
    message("Python version: ", py_config$version)
    message("NumPy version: ", boom_env$np$`__version__`)
    
    invisible(NULL)
  }, error = function(e) {
    stop("Failed to import BOOM Python modules. Error: ", e$message)
  })
}

#' Check BOOM Installation
#' 
#' Verify that the BOOM Python backend is properly installed and accessible.
#' 
#' @return Logical indicating if BOOM is properly installed
#' @export
#' 
#' @examples
#' \dontrun{
#' boom_check_installation()
#' }
boom_check_installation <- function() {
  # Check if Python is available
  if (!reticulate::py_available()) {
    message("Python is not available")
    return(FALSE)
  }
  
  # Check if boom module can be imported
  if (!reticulate::py_module_available("boom")) {
    message("BOOM Python module not found")
    message("Make sure boom_py is in your Python path")
    return(FALSE)
  }
  
  # Try to import and test basic functionality
  tryCatch({
    boom <- reticulate::import("boom")
    vec <- boom$linalg$Vector(c(1, 2, 3))
    
    message("BOOM Python backend is properly installed")
    message("BOOM version: Python implementation")
    return(TRUE)
  }, error = function(e) {
    message("Error testing BOOM: ", e$message)
    return(FALSE)
  })
}

#' Get BOOM Version Information
#' 
#' @return List with version information
#' @export
boom_version <- function() {
  if (!exists("boom", envir = boom_env)) {
    boom_setup()
  }
  
  list(
    r_package_version = utils::packageVersion("boompy"),
    python_version = reticulate::py_config()$version,
    numpy_version = boom_env$np$`__version__`,
    backend = "Python"
  )
}