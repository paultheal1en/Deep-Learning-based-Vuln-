static int intel_hda_post_load(void *opaque, int version)

{

    IntelHDAState* d = opaque;

    int i;



    dprint(d, 1, "%s\n", __FUNCTION__);

    for (i = 0; i < ARRAY_SIZE(d->st); i++) {

        if (d->st[i].ctl & 0x02) {

            intel_hda_parse_bdl(d, &d->st[i]);

        }

    }

    intel_hda_update_irq(d);

    return 0;

}
