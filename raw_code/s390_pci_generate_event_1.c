static void s390_pci_generate_event(uint8_t cc, uint16_t pec, uint32_t fh,

                                    uint32_t fid, uint64_t faddr, uint32_t e)

{

    SeiContainer *sei_cont = g_malloc0(sizeof(SeiContainer));

    S390pciState *s = S390_PCI_HOST_BRIDGE(

        object_resolve_path(TYPE_S390_PCI_HOST_BRIDGE, NULL));



    if (!s) {

        return;

    }



    sei_cont->fh = fh;

    sei_cont->fid = fid;

    sei_cont->cc = cc;

    sei_cont->pec = pec;

    sei_cont->faddr = faddr;

    sei_cont->e = e;



    QTAILQ_INSERT_TAIL(&s->pending_sei, sei_cont, link);

    css_generate_css_crws(0);

}
