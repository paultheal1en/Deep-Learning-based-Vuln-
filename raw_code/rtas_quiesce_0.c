static void rtas_quiesce(sPAPREnvironment *spapr, uint32_t token,

                         uint32_t nargs, target_ulong args,

                         uint32_t nret, target_ulong rets)

{

    VIOsPAPRBus *bus = spapr->vio_bus;

    BusChild *kid;

    VIOsPAPRDevice *dev = NULL;



    if (nargs != 0) {

        rtas_st(rets, 0, -3);

        return;

    }



    QTAILQ_FOREACH(kid, &bus->bus.children, sibling) {

        dev = (VIOsPAPRDevice *)kid->child;

        spapr_vio_quiesce_one(dev);

    }



    rtas_st(rets, 0, 0);

}
