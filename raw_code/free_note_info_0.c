static void free_note_info(struct elf_note_info *info)

{

    struct elf_thread_status *ets;



    while (!TAILQ_EMPTY(&info->thread_list)) {

        ets = TAILQ_FIRST(&info->thread_list);

        TAILQ_REMOVE(&info->thread_list, ets, ets_link);

        qemu_free(ets);

    }



    qemu_free(info->prstatus);

    qemu_free(info->psinfo);

    qemu_free(info->notes);

}
