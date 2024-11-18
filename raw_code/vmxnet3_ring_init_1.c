static inline void vmxnet3_ring_init(Vmxnet3Ring *ring,

                                     hwaddr pa,

                                     size_t size,

                                     size_t cell_size,

                                     bool zero_region)

{

    ring->pa = pa;

    ring->size = size;

    ring->cell_size = cell_size;

    ring->gen = VMXNET3_INIT_GEN;

    ring->next = 0;



    if (zero_region) {

        vmw_shmem_set(pa, 0, size * cell_size);

    }

}
