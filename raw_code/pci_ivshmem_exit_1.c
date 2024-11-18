static void pci_ivshmem_exit(PCIDevice *dev)

{

    IVShmemState *s = IVSHMEM(dev);

    int i;



    fifo8_destroy(&s->incoming_fifo);



    if (s->migration_blocker) {

        migrate_del_blocker(s->migration_blocker);

        error_free(s->migration_blocker);

    }



    if (s->shm_fd >= 0) {

        void *addr = memory_region_get_ram_ptr(&s->ivshmem);



        vmstate_unregister_ram(&s->ivshmem, DEVICE(dev));

        memory_region_del_subregion(&s->bar, &s->ivshmem);



        if (munmap(addr, s->ivshmem_size) == -1) {

            error_report("Failed to munmap shared memory %s", strerror(errno));

        }

    }



    if (s->eventfd_chr) {

        for (i = 0; i < s->vectors; i++) {

            if (s->eventfd_chr[i]) {

                qemu_chr_free(s->eventfd_chr[i]);

            }

        }

        g_free(s->eventfd_chr);

    }



    if (s->peers) {

        for (i = 0; i < s->nb_peers; i++) {

            close_peer_eventfds(s, i);

        }

        g_free(s->peers);

    }



    if (ivshmem_has_feature(s, IVSHMEM_MSI)) {

        msix_uninit_exclusive_bar(dev);

    }



    g_free(s->eventfd_table);

}
