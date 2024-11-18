static uint8_t lsi_reg_readb(LSIState *s, int offset)

{

    uint8_t tmp;

#define CASE_GET_REG32(name, addr) \

    case addr: return s->name & 0xff; \

    case addr + 1: return (s->name >> 8) & 0xff; \

    case addr + 2: return (s->name >> 16) & 0xff; \

    case addr + 3: return (s->name >> 24) & 0xff;



#ifdef DEBUG_LSI_REG

    DPRINTF("Read reg %x\n", offset);

#endif

    switch (offset) {

    case 0x00: /* SCNTL0 */

        return s->scntl0;

    case 0x01: /* SCNTL1 */

        return s->scntl1;

    case 0x02: /* SCNTL2 */

        return s->scntl2;

    case 0x03: /* SCNTL3 */

        return s->scntl3;

    case 0x04: /* SCID */

        return s->scid;

    case 0x05: /* SXFER */

        return s->sxfer;

    case 0x06: /* SDID */

        return s->sdid;

    case 0x07: /* GPREG0 */

        return 0x7f;

    case 0x08: /* Revision ID */

        return 0x00;

    case 0xa: /* SSID */

        return s->ssid;

    case 0xb: /* SBCL */

        /* ??? This is not correct. However it's (hopefully) only

           used for diagnostics, so should be ok.  */

        return 0;

    case 0xc: /* DSTAT */

        tmp = s->dstat | 0x80;

        if ((s->istat0 & LSI_ISTAT0_INTF) == 0)

            s->dstat = 0;

        lsi_update_irq(s);

        return tmp;

    case 0x0d: /* SSTAT0 */

        return s->sstat0;

    case 0x0e: /* SSTAT1 */

        return s->sstat1;

    case 0x0f: /* SSTAT2 */

        return s->scntl1 & LSI_SCNTL1_CON ? 0 : 2;

    CASE_GET_REG32(dsa, 0x10)

    case 0x14: /* ISTAT0 */

        return s->istat0;

    case 0x16: /* MBOX0 */

        return s->mbox0;

    case 0x17: /* MBOX1 */

        return s->mbox1;

    case 0x18: /* CTEST0 */

        return 0xff;

    case 0x19: /* CTEST1 */

        return 0;

    case 0x1a: /* CTEST2 */

        tmp = LSI_CTEST2_DACK | LSI_CTEST2_CM;

        if (s->istat0 & LSI_ISTAT0_SIGP) {

            s->istat0 &= ~LSI_ISTAT0_SIGP;

            tmp |= LSI_CTEST2_SIGP;

        }

        return tmp;

    case 0x1b: /* CTEST3 */

        return s->ctest3;

    CASE_GET_REG32(temp, 0x1c)

    case 0x20: /* DFIFO */

        return 0;

    case 0x21: /* CTEST4 */

        return s->ctest4;

    case 0x22: /* CTEST5 */

        return s->ctest5;

    case 0x23: /* CTEST6 */

         return 0;

    case 0x24: /* DBC[0:7] */

        return s->dbc & 0xff;

    case 0x25: /* DBC[8:15] */

        return (s->dbc >> 8) & 0xff;

    case 0x26: /* DBC[16->23] */

        return (s->dbc >> 16) & 0xff;

    case 0x27: /* DCMD */

        return s->dcmd;

    CASE_GET_REG32(dsp, 0x2c)

    CASE_GET_REG32(dsps, 0x30)

    CASE_GET_REG32(scratch[0], 0x34)

    case 0x38: /* DMODE */

        return s->dmode;

    case 0x39: /* DIEN */

        return s->dien;

    case 0x3b: /* DCNTL */

        return s->dcntl;

    case 0x40: /* SIEN0 */

        return s->sien0;

    case 0x41: /* SIEN1 */

        return s->sien1;

    case 0x42: /* SIST0 */

        tmp = s->sist0;

        s->sist0 = 0;

        lsi_update_irq(s);

        return tmp;

    case 0x43: /* SIST1 */

        tmp = s->sist1;

        s->sist1 = 0;

        lsi_update_irq(s);

        return tmp;

    case 0x47: /* GPCNTL0 */

        return 0x0f;

    case 0x48: /* STIME0 */

        return s->stime0;

    case 0x4a: /* RESPID0 */

        return s->respid0;

    case 0x4b: /* RESPID1 */

        return s->respid1;

    case 0x4d: /* STEST1 */

        return s->stest1;

    case 0x4e: /* STEST2 */

        return s->stest2;

    case 0x4f: /* STEST3 */

        return s->stest3;

    case 0x50: /* SIDL */

        /* This is needed by the linux drivers.  We currently only update it

           during the MSG IN phase.  */

        return s->sidl;

    case 0x52: /* STEST4 */

        return 0xe0;

    case 0x56: /* CCNTL0 */

        return s->ccntl0;

    case 0x57: /* CCNTL1 */

        return s->ccntl1;

    case 0x58: /* SBDL */

        /* Some drivers peek at the data bus during the MSG IN phase.  */

        if ((s->sstat1 & PHASE_MASK) == PHASE_MI)

            return s->msg[0];

        return 0;

    case 0x59: /* SBDL high */

        return 0;

    CASE_GET_REG32(mmrs, 0xa0)

    CASE_GET_REG32(mmws, 0xa4)

    CASE_GET_REG32(sfs, 0xa8)

    CASE_GET_REG32(drs, 0xac)

    CASE_GET_REG32(sbms, 0xb0)

    CASE_GET_REG32(dmbs, 0xb4)

    CASE_GET_REG32(dnad64, 0xb8)

    CASE_GET_REG32(pmjad1, 0xc0)

    CASE_GET_REG32(pmjad2, 0xc4)

    CASE_GET_REG32(rbc, 0xc8)

    CASE_GET_REG32(ua, 0xcc)

    CASE_GET_REG32(ia, 0xd4)

    CASE_GET_REG32(sbc, 0xd8)

    CASE_GET_REG32(csbc, 0xdc)

    }

    if (offset >= 0x5c && offset < 0xa0) {

        int n;

        int shift;

        n = (offset - 0x58) >> 2;

        shift = (offset & 3) * 8;

        return (s->scratch[n] >> shift) & 0xff;

    }

    BADF("readb 0x%x\n", offset);

    exit(1);

#undef CASE_GET_REG32

}
