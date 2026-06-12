C++ API Reference
=================

The C++ reference is generated from Doxygen comments in ``src/cpp`` and exposed
through Sphinx with Breathe. It focuses on public data structures, solver entry
points, output/provenance helpers, and geometry utilities used by the simulator.

Core Types
----------

.. doxygenstruct:: fmmgalaxy::Vec3
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::Particle
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::PhysicsParams
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::Diagnostics
   :project: fmmgalaxy
   :members:

.. doxygenfunction:: fmmgalaxy::build_summary
   :project: fmmgalaxy

Configuration and Provenance
----------------------------

.. doxygenenum:: fmmgalaxy::OutputFormat
   :project: fmmgalaxy

.. doxygenstruct:: fmmgalaxy::OutputConfig
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::SimulationConfig
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::GalaxyConfig
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::RunProvenance
   :project: fmmgalaxy
   :members:

.. doxygenfunction:: fmmgalaxy::parse_output_format
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::output_format_name
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::default_config
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::load_config
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::collect_run_provenance
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::sha256_file
   :project: fmmgalaxy

Solvers
-------

.. doxygenclass:: fmmgalaxy::BarnesHutSolver
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FmmOptions
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FmmStats
   :project: fmmgalaxy
   :members:

.. doxygenclass:: fmmgalaxy::FastMultipoleSolver
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::CudaTreeOptions
   :project: fmmgalaxy
   :members:

.. doxygenfunction:: fmmgalaxy::softened_acceleration
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_direct_accelerations
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_tree_accelerations
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_fmm_accelerations
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_cuda_direct_accelerations
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_cuda_tree_accelerations
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::compute_cuda_fmm_accelerations
   :project: fmmgalaxy

Tree and FMM Data
-----------------

.. doxygenstruct:: fmmgalaxy::CartesianMoments
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::LocalExpansion
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FlatTreeNode
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FlatTreeData
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FlatFmmLeaf
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::FlatFmmData
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::TreeRootCube
   :project: fmmgalaxy
   :members:

.. doxygenfunction:: fmmgalaxy::root_cube_for_particles
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::child_index_for_position
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::child_center
   :project: fmmgalaxy

MPI and I/O
-----------

.. doxygenstruct:: fmmgalaxy::MpiExecution
   :project: fmmgalaxy
   :members:

.. doxygenstruct:: fmmgalaxy::OwnershipRange
   :project: fmmgalaxy
   :members:

.. doxygenclass:: fmmgalaxy::SnapshotWriter
   :project: fmmgalaxy
   :members:

.. doxygenclass:: fmmgalaxy::ParquetConverter
   :project: fmmgalaxy
   :members:

.. doxygenfunction:: fmmgalaxy::mpi_execution
   :project: fmmgalaxy

.. doxygenfunction:: fmmgalaxy::ownership_for_rank
   :project: fmmgalaxy
