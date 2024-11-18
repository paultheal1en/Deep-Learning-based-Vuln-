static void balloon_stats_get_all(Object *obj, Visitor *v, const char *name,

                                  void *opaque, Error **errp)

{

    Error *err = NULL;

    VirtIOBalloon *s = opaque;

    int i;



    visit_start_struct(v, name, NULL, 0, &err);

    if (err) {

        goto out;

    }

    visit_type_int(v, "last-update", &s->stats_last_update, &err);

    if (err) {

        goto out_end;

    }



    visit_start_struct(v, "stats", NULL, 0, &err);

    if (err) {

        goto out_end;

    }

    for (i = 0; i < VIRTIO_BALLOON_S_NR; i++) {

        visit_type_uint64(v, balloon_stat_names[i], &s->stats[i], &err);

        if (err) {

            break;

        }

    }

    error_propagate(errp, err);

    err = NULL;

    visit_end_struct(v, &err);



out_end:

    error_propagate(errp, err);

    err = NULL;

    visit_end_struct(v, &err);

out:

    error_propagate(errp, err);

}
