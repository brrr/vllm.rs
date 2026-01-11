//! BARM Worker module for distributed model serving via Zenoh
//!
//! This module provides the worker-side functionality for the BARM distributed
//! inference system, including Zenoh communication and weight loading.

pub mod zenoh_client;
pub mod weight_loader;

pub use zenoh_client::ZenohClient;
pub use weight_loader::WeightLoader;
