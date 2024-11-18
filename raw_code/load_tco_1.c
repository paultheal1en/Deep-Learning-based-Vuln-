static void load_tco(const TestData *d)

{

    qpci_io_writew(d->dev, d->tco_io_base + TCO_RLD, 4);

}
