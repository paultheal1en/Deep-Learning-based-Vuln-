static inline int ohci_put_ed(OHCIState *ohci,

                              uint32_t addr, struct ohci_ed *ed)

{

    return put_dwords(ohci, addr, (uint32_t *)ed, sizeof(*ed) >> 2);

}
