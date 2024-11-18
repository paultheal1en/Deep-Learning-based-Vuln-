static void qed_unplug_allocating_write_reqs(BDRVQEDState *s)

{

    assert(s->allocating_write_reqs_plugged);



    s->allocating_write_reqs_plugged = false;

    qemu_co_enter_next(&s->allocating_write_reqs);

}
