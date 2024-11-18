static inline void vmxnet3_ring_read_curr_cell(Vmxnet3Ring *ring, void *buff)

{

    vmw_shmem_read(vmxnet3_ring_curr_cell_pa(ring), buff, ring->cell_size);

}
