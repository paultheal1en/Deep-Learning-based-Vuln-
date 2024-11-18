static void lsi_soft_reset(LSIState *s)

{

    lsi_request *p;



    DPRINTF("Reset\n");

    s->carry = 0;



    s->msg_action = 0;

    s->msg_len = 0;

    s->waiting = 0;

    s->dsa = 0;

    s->dnad = 0;

    s->dbc = 0;

    s->temp = 0;

    memset(s->scratch, 0, sizeof(s->scratch));

    s->istat0 = 0;

    s->istat1 = 0;

    s->dcmd = 0x40;

    s->dstat = LSI_DSTAT_DFE;

    s->dien = 0;

    s->sist0 = 0;

    s->sist1 = 0;

    s->sien0 = 0;

    s->sien1 = 0;

    s->mbox0 = 0;

    s->mbox1 = 0;

    s->dfifo = 0;

    s->ctest2 = LSI_CTEST2_DACK;

    s->ctest3 = 0;

    s->ctest4 = 0;

    s->ctest5 = 0;

    s->ccntl0 = 0;

    s->ccntl1 = 0;

    s->dsp = 0;

    s->dsps = 0;

    s->dmode = 0;

    s->dcntl = 0;

    s->scntl0 = 0xc0;

    s->scntl1 = 0;

    s->scntl2 = 0;

    s->scntl3 = 0;

    s->sstat0 = 0;

    s->sstat1 = 0;

    s->scid = 7;

    s->sxfer = 0;

    s->socl = 0;

    s->sdid = 0;

    s->ssid = 0;

    s->stest1 = 0;

    s->stest2 = 0;

    s->stest3 = 0;

    s->sidl = 0;

    s->stime0 = 0;

    s->respid0 = 0x80;

    s->respid1 = 0;

    s->mmrs = 0;

    s->mmws = 0;

    s->sfs = 0;

    s->drs = 0;

    s->sbms = 0;

    s->dbms = 0;

    s->dnad64 = 0;

    s->pmjad1 = 0;

    s->pmjad2 = 0;

    s->rbc = 0;

    s->ua = 0;

    s->ia = 0;

    s->sbc = 0;

    s->csbc = 0;

    s->sbr = 0;

    while (!QTAILQ_EMPTY(&s->queue)) {

        p = QTAILQ_FIRST(&s->queue);

        QTAILQ_REMOVE(&s->queue, p, next);

        g_free(p);

    }

    if (s->current) {

        g_free(s->current);

        s->current = NULL;

    }

}
