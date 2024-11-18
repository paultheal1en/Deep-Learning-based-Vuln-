static PCIDevice *find_dev(sPAPREnvironment *spapr,

                           uint64_t buid, uint32_t config_addr)

{

    int devfn = (config_addr >> 8) & 0xFF;

    sPAPRPHBState *phb;



    QLIST_FOREACH(phb, &spapr->phbs, list) {

        BusChild *kid;



        if (phb->buid != buid) {

            continue;

        }



        QTAILQ_FOREACH(kid, &phb->host_state.bus->qbus.children, sibling) {

            PCIDevice *dev = (PCIDevice *)kid->child;

            if (dev->devfn == devfn) {

                return dev;

            }

        }

    }



    return NULL;

}
