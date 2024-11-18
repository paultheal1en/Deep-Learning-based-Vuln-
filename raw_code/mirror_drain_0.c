static void mirror_drain(MirrorBlockJob *s)

{

    while (s->in_flight > 0) {

        mirror_wait_for_io(s);

    }

}
