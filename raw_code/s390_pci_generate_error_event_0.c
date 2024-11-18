static void s390_pci_generate_error_event(uint16_t pec, uint32_t fh,

                                          uint32_t fid, uint64_t faddr,

                                          uint32_t e)

{

    s390_pci_generate_event(1, pec, fh, fid, faddr, e);

}
