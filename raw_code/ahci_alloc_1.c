static uint64_t ahci_alloc(AHCIQState *ahci, size_t bytes)

{

    return qmalloc(ahci->parent, bytes);

}
