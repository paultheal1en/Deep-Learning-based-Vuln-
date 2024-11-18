static void colo_compare_finalize(Object *obj)

{

    CompareState *s = COLO_COMPARE(obj);



    qemu_chr_fe_deinit(&s->chr_pri_in, false);

    qemu_chr_fe_deinit(&s->chr_sec_in, false);

    qemu_chr_fe_deinit(&s->chr_out, false);



    g_main_loop_quit(s->compare_loop);

    qemu_thread_join(&s->thread);



    /* Release all unhandled packets after compare thead exited */

    g_queue_foreach(&s->conn_list, colo_flush_packets, s);



    g_queue_clear(&s->conn_list);



    g_hash_table_destroy(s->connection_track_table);

    g_free(s->pri_indev);

    g_free(s->sec_indev);

    g_free(s->outdev);

}
