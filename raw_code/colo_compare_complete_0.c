static void colo_compare_complete(UserCreatable *uc, Error **errp)

{

    CompareState *s = COLO_COMPARE(uc);

    Chardev *chr;

    char thread_name[64];

    static int compare_id;



    if (!s->pri_indev || !s->sec_indev || !s->outdev) {

        error_setg(errp, "colo compare needs 'primary_in' ,"

                   "'secondary_in','outdev' property set");

        return;

    } else if (!strcmp(s->pri_indev, s->outdev) ||

               !strcmp(s->sec_indev, s->outdev) ||

               !strcmp(s->pri_indev, s->sec_indev)) {

        error_setg(errp, "'indev' and 'outdev' could not be same "

                   "for compare module");

        return;

    }



    if (find_and_check_chardev(&chr, s->pri_indev, errp) ||

        !qemu_chr_fe_init(&s->chr_pri_in, chr, errp)) {

        return;

    }



    if (find_and_check_chardev(&chr, s->sec_indev, errp) ||

        !qemu_chr_fe_init(&s->chr_sec_in, chr, errp)) {

        return;

    }



    if (find_and_check_chardev(&chr, s->outdev, errp) ||

        !qemu_chr_fe_init(&s->chr_out, chr, errp)) {

        return;

    }



    net_socket_rs_init(&s->pri_rs, compare_pri_rs_finalize, s->vnet_hdr);

    net_socket_rs_init(&s->sec_rs, compare_sec_rs_finalize, s->vnet_hdr);



    g_queue_init(&s->conn_list);



    s->connection_track_table = g_hash_table_new_full(connection_key_hash,

                                                      connection_key_equal,

                                                      g_free,

                                                      connection_destroy);



    sprintf(thread_name, "colo-compare %d", compare_id);

    qemu_thread_create(&s->thread, thread_name,

                       colo_compare_thread, s,

                       QEMU_THREAD_JOINABLE);

    compare_id++;



    return;

}
