void xen_pt_msi_disable(XenPCIPassthroughState *s)

{

    XenPTMSI *msi = s->msi;



    if (!msi) {

        return;

    }



    xen_pt_msi_set_enable(s, false);



    msi_msix_disable(s, msi_addr64(msi), msi->data, msi->pirq, false,

                     msi->initialized);



    /* clear msi info */

    msi->flags = 0;

    msi->mapped = false;

    msi->pirq = XEN_PT_UNASSIGNED_PIRQ;

}
