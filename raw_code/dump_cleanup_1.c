static int dump_cleanup(DumpState *s)

{

    int ret = 0;



    guest_phys_blocks_free(&s->guest_phys_blocks);

    memory_mapping_list_free(&s->list);

    if (s->fd != -1) {

        close(s->fd);

    }

    if (s->resume) {

        vm_start();

    }



    return ret;

}
