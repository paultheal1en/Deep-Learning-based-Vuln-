int chsc_sei_nt2_get_event(void *res)

{

    ChscSeiNt2Res *nt2_res = (ChscSeiNt2Res *)res;

    PciCcdfAvail *accdf;

    PciCcdfErr *eccdf;

    int rc = 1;

    SeiContainer *sei_cont;

    S390pciState *s = S390_PCI_HOST_BRIDGE(

        object_resolve_path(TYPE_S390_PCI_HOST_BRIDGE, NULL));



    if (!s) {

        return rc;

    }



    sei_cont = QTAILQ_FIRST(&s->pending_sei);

    if (sei_cont) {

        QTAILQ_REMOVE(&s->pending_sei, sei_cont, link);

        nt2_res->nt = 2;

        nt2_res->cc = sei_cont->cc;

        nt2_res->length = cpu_to_be16(sizeof(ChscSeiNt2Res));

        switch (sei_cont->cc) {

        case 1: /* error event */

            eccdf = (PciCcdfErr *)nt2_res->ccdf;

            eccdf->fid = cpu_to_be32(sei_cont->fid);

            eccdf->fh = cpu_to_be32(sei_cont->fh);

            eccdf->e = cpu_to_be32(sei_cont->e);

            eccdf->faddr = cpu_to_be64(sei_cont->faddr);

            eccdf->pec = cpu_to_be16(sei_cont->pec);

            break;

        case 2: /* availability event */

            accdf = (PciCcdfAvail *)nt2_res->ccdf;

            accdf->fid = cpu_to_be32(sei_cont->fid);

            accdf->fh = cpu_to_be32(sei_cont->fh);

            accdf->pec = cpu_to_be16(sei_cont->pec);

            break;

        default:

            abort();

        }

        g_free(sei_cont);

        rc = 0;

    }



    return rc;

}
