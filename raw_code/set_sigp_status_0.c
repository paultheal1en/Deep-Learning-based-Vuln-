static void set_sigp_status(SigpInfo *si, uint64_t status)

{

    *si->status_reg &= 0xffffffff00000000ULL;

    *si->status_reg |= status;

    si->cc = SIGP_CC_STATUS_STORED;

}
