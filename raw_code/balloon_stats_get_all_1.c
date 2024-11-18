static void balloon_stats_get_all(Object *obj, struct Visitor *v,

                                  void *opaque, const char *name, Error **errp)

{

    Error *err = NULL;

    VirtIOBalloon *s = opaque;

    int i;



    if (!s->stats_last_update) {

        error_setg(errp, "guest hasn't updated any stats yet");

        return;

    }



    visit_start_struct(v, NULL, "guest-stats", name, 0, &err);

    if (err) {

        goto out;

    }



    visit_type_int(v, &s->stats_last_update, "last-update", &err);



    visit_start_struct(v, NULL, NULL, "stats", 0, &err);

    if (err) {

        goto out_end;

    }

        

    for (i = 0; i < VIRTIO_BALLOON_S_NR; i++) {

        visit_type_int64(v, (int64_t *) &s->stats[i], balloon_stat_names[i],

                         &err);

    }

    visit_end_struct(v, &err);



out_end:

    visit_end_struct(v, &err);



out:

    error_propagate(errp, err);

}
