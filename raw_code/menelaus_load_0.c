static int menelaus_load(QEMUFile *f, void *opaque, int version_id)

{

    MenelausState *s = (MenelausState *) opaque;



    s->firstbyte = qemu_get_be32(f);

    qemu_get_8s(f, &s->reg);



    if (s->rtc.ctrl & 1)					/* RTC_EN */

        menelaus_rtc_stop(s);

    qemu_get_8s(f, &s->vcore[0]);

    qemu_get_8s(f, &s->vcore[1]);

    qemu_get_8s(f, &s->vcore[2]);

    qemu_get_8s(f, &s->vcore[3]);

    qemu_get_8s(f, &s->vcore[4]);

    qemu_get_8s(f, &s->dcdc[0]);

    qemu_get_8s(f, &s->dcdc[1]);

    qemu_get_8s(f, &s->dcdc[2]);

    qemu_get_8s(f, &s->ldo[0]);

    qemu_get_8s(f, &s->ldo[1]);

    qemu_get_8s(f, &s->ldo[2]);

    qemu_get_8s(f, &s->ldo[3]);

    qemu_get_8s(f, &s->ldo[4]);

    qemu_get_8s(f, &s->ldo[5]);

    qemu_get_8s(f, &s->ldo[6]);

    qemu_get_8s(f, &s->ldo[7]);

    qemu_get_8s(f, &s->sleep[0]);

    qemu_get_8s(f, &s->sleep[1]);

    qemu_get_8s(f, &s->osc);

    qemu_get_8s(f, &s->detect);

    qemu_get_be16s(f, &s->mask);

    qemu_get_be16s(f, &s->status);

    qemu_get_8s(f, &s->dir);

    qemu_get_8s(f, &s->inputs);

    qemu_get_8s(f, &s->outputs);

    qemu_get_8s(f, &s->bbsms);

    qemu_get_8s(f, &s->pull[0]);

    qemu_get_8s(f, &s->pull[1]);

    qemu_get_8s(f, &s->pull[2]);

    qemu_get_8s(f, &s->pull[3]);

    qemu_get_8s(f, &s->mmc_ctrl[0]);

    qemu_get_8s(f, &s->mmc_ctrl[1]);

    qemu_get_8s(f, &s->mmc_ctrl[2]);

    qemu_get_8s(f, &s->mmc_debounce);

    qemu_get_8s(f, &s->rtc.ctrl);

    qemu_get_be16s(f, &s->rtc.comp);

    s->rtc.next = qemu_get_be16(f);

    tm_get(f, &s->rtc.new);

    tm_get(f, &s->rtc.alm);

    s->pwrbtn_state = qemu_get_byte(f);

    menelaus_alm_update(s);

    menelaus_update(s);

    if (s->rtc.ctrl & 1)					/* RTC_EN */

        menelaus_rtc_start(s);



    i2c_slave_load(f, &s->i2c);

    return 0;

}
