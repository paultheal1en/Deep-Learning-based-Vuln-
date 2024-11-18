static int ivshmem_setup_interrupts(IVShmemState *s)

{

    /* allocate QEMU callback data for receiving interrupts */

    s->msi_vectors = g_malloc0(s->vectors * sizeof(MSIVector));



    if (ivshmem_has_feature(s, IVSHMEM_MSI)) {

        if (msix_init_exclusive_bar(PCI_DEVICE(s), s->vectors, 1)) {

            return -1;

        }



        IVSHMEM_DPRINTF("msix initialized (%d vectors)\n", s->vectors);

        ivshmem_msix_vector_use(s);

    }



    return 0;

}
