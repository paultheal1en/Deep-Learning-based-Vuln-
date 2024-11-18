static void menelaus_save(QEMUFile *f, void *opaque)

{

    MenelausState *s = (MenelausState *) opaque;



    qemu_put_be32(f, s->firstbyte);

    qemu_put_8s(f, &s->reg);



    qemu_put_8s(f, &s->vcore[0]);

    qemu_put_8s(f, &s->vcore[1]);

    qemu_put_8s(f, &s->vcore[2]);

    qemu_put_8s(f, &s->vcore[3]);

    qemu_put_8s(f, &s->vcore[4]);

    qemu_put_8s(f, &s->dcdc[0]);

    qemu_put_8s(f, &s->dcdc[1]);

    qemu_put_8s(f, &s->dcdc[2]);

    qemu_put_8s(f, &s->ldo[0]);

    qemu_put_8s(f, &s->ldo[1]);

    qemu_put_8s(f, &s->ldo[2]);

    qemu_put_8s(f, &s->ldo[3]);

    qemu_put_8s(f, &s->ldo[4]);

    qemu_put_8s(f, &s->ldo[5]);

    qemu_put_8s(f, &s->ldo[6]);

    qemu_put_8s(f, &s->ldo[7]);

    qemu_put_8s(f, &s->sleep[0]);

    qemu_put_8s(f, &s->sleep[1]);

    qemu_put_8s(f, &s->osc);

    qemu_put_8s(f, &s->detect);

    qemu_put_be16s(f, &s->mask);

    qemu_put_be16s(f, &s->status);

    qemu_put_8s(f, &s->dir);

    qemu_put_8s(f, &s->inputs);

    qemu_put_8s(f, &s->outputs);

    qemu_put_8s(f, &s->bbsms);

    qemu_put_8s(f, &s->pull[0]);

    qemu_put_8s(f, &s->pull[1]);

    qemu_put_8s(f, &s->pull[2]);

    qemu_put_8s(f, &s->pull[3]);

    qemu_put_8s(f, &s->mmc_ctrl[0]);

    qemu_put_8s(f, &s->mmc_ctrl[1]);

    qemu_put_8s(f, &s->mmc_ctrl[2]);

    qemu_put_8s(f, &s->mmc_debounce);

    qemu_put_8s(f, &s->rtc.ctrl);

    qemu_put_be16s(f, &s->rtc.comp);

    /* Should be <= 1000 */

    qemu_put_be16(f, s->rtc.next - qemu_get_clock(rt_clock));

    tm_put(f, &s->rtc.new);

    tm_put(f, &s->rtc.alm);

    qemu_put_byte(f, s->pwrbtn_state);



    i2c_slave_save(f, &s->i2c);

}
