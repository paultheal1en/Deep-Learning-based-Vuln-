static void arm_sysctl_write(void *opaque, target_phys_addr_t offset,

                             uint64_t val, unsigned size)

{

    arm_sysctl_state *s = (arm_sysctl_state *)opaque;



    switch (offset) {

    case 0x08: /* LED */

        s->leds = val;

    case 0x0c: /* OSC0 */

    case 0x10: /* OSC1 */

    case 0x14: /* OSC2 */

    case 0x18: /* OSC3 */

    case 0x1c: /* OSC4 */

        /* ??? */

        break;

    case 0x20: /* LOCK */

        if (val == LOCK_VALUE)

            s->lockval = val;

        else

            s->lockval = val & 0x7fff;

        break;

    case 0x28: /* CFGDATA1 */

        /* ??? Need to implement this.  */

        s->cfgdata1 = val;

        break;

    case 0x2c: /* CFGDATA2 */

        /* ??? Need to implement this.  */

        s->cfgdata2 = val;

        break;

    case 0x30: /* FLAGSSET */

        s->flags |= val;

        break;

    case 0x34: /* FLAGSCLR */

        s->flags &= ~val;

        break;

    case 0x38: /* NVFLAGSSET */

        s->nvflags |= val;

        break;

    case 0x3c: /* NVFLAGSCLR */

        s->nvflags &= ~val;

        break;

    case 0x40: /* RESETCTL */

        switch (board_id(s)) {

        case BOARD_ID_PB926:

            if (s->lockval == LOCK_VALUE) {

                s->resetlevel = val;

                if (val & 0x100) {

                    qemu_system_reset_request();

                }

            }

            break;

        case BOARD_ID_PBX:

        case BOARD_ID_PBA8:

            if (s->lockval == LOCK_VALUE) {

                s->resetlevel = val;

                if (val & 0x04) {

                    qemu_system_reset_request();

                }

            }

            break;

        case BOARD_ID_VEXPRESS:

        case BOARD_ID_EB:

        default:

            /* reserved: RAZ/WI */

            break;

        }

        break;

    case 0x44: /* PCICTL */

        /* nothing to do.  */

        break;

    case 0x4c: /* FLASH */

        break;

    case 0x50: /* CLCD */

        switch (board_id(s)) {

        case BOARD_ID_PB926:

            /* On 926 bits 13:8 are R/O, bits 1:0 control

             * the mux that defines how to interpret the PL110

             * graphics format, and other bits are r/w but we

             * don't implement them to do anything.

             */

            s->sys_clcd &= 0x3f00;

            s->sys_clcd |= val & ~0x3f00;

            qemu_set_irq(s->pl110_mux_ctrl, val & 3);

            break;

        case BOARD_ID_EB:

            /* The EB is the same except that there is no mux since

             * the EB has a PL111.

             */

            s->sys_clcd &= 0x3f00;

            s->sys_clcd |= val & ~0x3f00;

            break;

        case BOARD_ID_PBA8:

        case BOARD_ID_PBX:

            /* On PBA8 and PBX bit 7 is r/w and all other bits

             * are either r/o or RAZ/WI.

             */

            s->sys_clcd &= (1 << 7);

            s->sys_clcd |= val & ~(1 << 7);

            break;

        case BOARD_ID_VEXPRESS:

        default:

            /* On VExpress this register is unimplemented and will RAZ/WI */

            break;

        }

    case 0x54: /* CLCDSER */

    case 0x64: /* DMAPSR0 */

    case 0x68: /* DMAPSR1 */

    case 0x6c: /* DMAPSR2 */

    case 0x70: /* IOSEL */

    case 0x74: /* PLDCTL */

    case 0x80: /* BUSID */

    case 0x84: /* PROCID0 */

    case 0x88: /* PROCID1 */

    case 0x8c: /* OSCRESET0 */

    case 0x90: /* OSCRESET1 */

    case 0x94: /* OSCRESET2 */

    case 0x98: /* OSCRESET3 */

    case 0x9c: /* OSCRESET4 */

        break;

    case 0xa0: /* SYS_CFGDATA */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        s->sys_cfgdata = val;

        return;

    case 0xa4: /* SYS_CFGCTRL */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        s->sys_cfgctrl = val & ~(3 << 18);

        s->sys_cfgstat = 1;            /* complete */

        switch (s->sys_cfgctrl) {

        case 0xc0800000:            /* SYS_CFG_SHUTDOWN to motherboard */

            qemu_system_shutdown_request();

            break;

        case 0xc0900000:            /* SYS_CFG_REBOOT to motherboard */

            qemu_system_reset_request();

            break;

        default:

            s->sys_cfgstat |= 2;        /* error */

        }

        return;

    case 0xa8: /* SYS_CFGSTAT */

        if (board_id(s) != BOARD_ID_VEXPRESS) {

            goto bad_reg;

        }

        s->sys_cfgstat = val & 3;

        return;

    default:

    bad_reg:

        printf ("arm_sysctl_write: Bad register offset 0x%x\n", (int)offset);

        return;

    }

}
