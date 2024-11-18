static void fill_prefetch_fifo(struct omap_gpmc_s *s)
{
    /* Fill the prefetch FIFO by reading data from NAND.
     * We do this synchronously, unlike the hardware which
     * will do this asynchronously. We refill when the
     * FIFO has THRESHOLD bytes free, and we always refill
     * as much data as possible starting at the top end
     * of the FIFO.
     * (We have to refill at THRESHOLD rather than waiting
     * for the FIFO to empty to allow for the case where
     * the FIFO size isn't an exact multiple of THRESHOLD
     * and we're doing DMA transfers.)
     * This means we never need to handle wrap-around in
     * the fifo-reading code, and the next byte of data
     * to read is always fifo[63 - fifopointer].
     */
    int fptr;
    int cs = prefetch_cs(s->prefetch.config1);
    int is16bit = (((s->cs_file[cs].config[0] >> 12) & 3) != 0);
    int bytes;
    /* Don't believe the bit of the OMAP TRM that says that COUNTVALUE
     * and TRANSFERCOUNT are in units of 16 bit words for 16 bit NAND.
     * Instead believe the bit that says it is always a byte count.
     */
    bytes = 64 - s->prefetch.fifopointer;
    if (bytes > s->prefetch.count) {
        bytes = s->prefetch.count;
    s->prefetch.count -= bytes;
    s->prefetch.fifopointer += bytes;
    fptr = 64 - s->prefetch.fifopointer;
    /* Move the existing data in the FIFO so it sits just
     * before what we're about to read in
     */
    while (fptr < (64 - bytes)) {
        s->prefetch.fifo[fptr] = s->prefetch.fifo[fptr + bytes];
        fptr++;
    while (fptr < 64) {
            uint32_t v = omap_nand_read(&s->cs_file[cs], 0, 2);
            s->prefetch.fifo[fptr++] = v & 0xff;
            s->prefetch.fifo[fptr++] = (v >> 8) & 0xff;
        } else {
            s->prefetch.fifo[fptr++] = omap_nand_read(&s->cs_file[cs], 0, 1);
    if (s->prefetch.startengine && (s->prefetch.count == 0)) {
        /* This was the final transfer: raise TERMINALCOUNTSTATUS */
        s->irqst |= 2;
        s->prefetch.startengine = 0;
    /* If there are any bytes in the FIFO at this point then
     * we must raise a DMA request (either this is a final part
     * transfer, or we filled the FIFO in which case we certainly
     * have THRESHOLD bytes available)
     */
    if (s->prefetch.fifopointer != 0) {
        omap_gpmc_dma_update(s, 1);
    omap_gpmc_int_update(s);