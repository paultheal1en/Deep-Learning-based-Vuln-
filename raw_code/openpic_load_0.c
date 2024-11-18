static int openpic_load(QEMUFile* f, void *opaque, int version_id)

{

    OpenPICState *opp = (OpenPICState *)opaque;

    unsigned int i;



    if (version_id != 1)

        return -EINVAL;



    qemu_get_be32s(f, &opp->glbc);

    qemu_get_be32s(f, &opp->veni);

    qemu_get_be32s(f, &opp->pint);

    qemu_get_be32s(f, &opp->spve);

    qemu_get_be32s(f, &opp->tifr);



    for (i = 0; i < opp->max_irq; i++) {

        qemu_get_be32s(f, &opp->src[i].ipvp);

        qemu_get_be32s(f, &opp->src[i].ide);

        qemu_get_sbe32s(f, &opp->src[i].last_cpu);

        qemu_get_sbe32s(f, &opp->src[i].pending);

    }



    qemu_get_be32s(f, &opp->nb_cpus);



    for (i = 0; i < opp->nb_cpus; i++) {

        qemu_get_be32s(f, &opp->dst[i].pctp);

        qemu_get_be32s(f, &opp->dst[i].pcsr);

        openpic_load_IRQ_queue(f, &opp->dst[i].raised);

        openpic_load_IRQ_queue(f, &opp->dst[i].servicing);

    }



    for (i = 0; i < MAX_TMR; i++) {

        qemu_get_be32s(f, &opp->timers[i].ticc);

        qemu_get_be32s(f, &opp->timers[i].tibc);

    }



    return 0;

}
