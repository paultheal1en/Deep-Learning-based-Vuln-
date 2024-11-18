static void do_stop_capture(Monitor *mon, const QDict *qdict)

{

    int i;

    int n = qdict_get_int(qdict, "n");

    CaptureState *s;



    for (s = capture_head.lh_first, i = 0; s; s = s->entries.le_next, ++i) {

        if (i == n) {

            s->ops.destroy (s->opaque);

            LIST_REMOVE (s, entries);

            qemu_free (s);

            return;

        }

    }

}
