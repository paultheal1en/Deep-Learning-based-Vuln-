static void colo_compare_finalize(Object *obj)

{

    CompareState *s = COLO_COMPARE(obj);



    qemu_chr_fe_deinit(&s->chr_pri_in);

    qemu_chr_fe_deinit(&s->chr_sec_in);

    qemu_chr_fe_deinit(&s->chr_out);



    g_queue_free(&s->conn_list);



    if (qemu_thread_is_self(&s->thread)) {

        /* compare connection */

        g_queue_foreach(&s->conn_list, colo_compare_connection, s);

        qemu_thread_join(&s->thread);

    }



    g_free(s->pri_indev);

    g_free(s->sec_indev);

    g_free(s->outdev);

}
