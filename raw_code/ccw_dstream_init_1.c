void ccw_dstream_init(CcwDataStream *cds, CCW1 const *ccw, ORB const *orb)
{
    /*
     * We don't support MIDA (an optional facility) yet and we
     * catch this earlier. Just for expressing the precondition.
     */
    g_assert(!(orb->ctrl1 & ORB_CTRL1_MASK_MIDAW));
    cds->flags = (orb->ctrl0 & ORB_CTRL0_MASK_I2K ? CDS_F_I2K : 0) |
                 (orb->ctrl0 & ORB_CTRL0_MASK_C64 ? CDS_F_C64 : 0) |
                 (ccw->flags & CCW_FLAG_IDA ? CDS_F_IDA : 0);
    cds->count = ccw->count;
    cds->cda_orig = ccw->cda;
    ccw_dstream_rewind(cds);
    if (!(cds->flags & CDS_F_IDA)) {
        cds->op_handler = ccw_dstream_rw_noflags;
    } else {
        assert(false);
    }
}