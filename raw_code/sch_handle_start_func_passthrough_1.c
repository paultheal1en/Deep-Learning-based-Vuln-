static int sch_handle_start_func_passthrough(SubchDev *sch)

{



    PMCW *p = &sch->curr_status.pmcw;

    SCSW *s = &sch->curr_status.scsw;

    int ret;



    ORB *orb = &sch->orb;

    if (!(s->ctrl & SCSW_ACTL_SUSP)) {

        assert(orb != NULL);

        p->intparm = orb->intparm;

    }



    /*

     * Only support prefetch enable mode.

     * Only support 64bit addressing idal.

     */

    if (!(orb->ctrl0 & ORB_CTRL0_MASK_PFCH) ||

        !(orb->ctrl0 & ORB_CTRL0_MASK_C64)) {

        return -EINVAL;

    }



    ret = s390_ccw_cmd_request(orb, s, sch->driver_data);

    switch (ret) {

    /* Currently we don't update control block and just return the cc code. */

    case 0:

        break;

    case -EBUSY:

        break;

    case -ENODEV:

        break;

    case -EACCES:

        /* Let's reflect an inaccessible host device by cc 3. */

        ret = -ENODEV;

        break;

    default:

       /*

        * All other return codes will trigger a program check,

        * or set cc to 1.

        */

       break;

    };



    return ret;

}
