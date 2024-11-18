static void set_tco_timeout(const TestData *d, uint16_t ticks)

{

    qpci_io_writew(d->dev, d->tco_io_base + TCO_TMR, ticks);

}
