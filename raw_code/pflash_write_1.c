static void pflash_write (pflash_t *pfl, target_ulong offset, uint32_t value,

                          int width)

{

    target_ulong boff;

    uint8_t *p;

    uint8_t cmd;



    /* WARNING: when the memory area is in ROMD mode, the offset is a

       ram offset, not a physical address */

    if (pfl->wcycle == 0)

        offset -= (target_ulong)(long)pfl->storage;

    else

        offset -= pfl->base;

        

    cmd = value;

    DPRINTF("%s: offset " TARGET_FMT_lx " %08x %d\n", __func__,

            offset, value, width);

    if (pfl->cmd != 0xA0 && cmd == 0xF0) {

        DPRINTF("%s: flash reset asked (%02x %02x)\n",

                __func__, pfl->cmd, cmd);

        goto reset_flash;

    }

    /* Set the device in I/O access mode */

    cpu_register_physical_memory(pfl->base, pfl->total_len, pfl->fl_mem);

    boff = offset & (pfl->sector_len - 1);

    if (pfl->width == 2)

        boff = boff >> 1;

    else if (pfl->width == 4)

        boff = boff >> 2;

    switch (pfl->wcycle) {

    case 0:

        /* We're in read mode */

    check_unlock0:

        if (boff == 0x55 && cmd == 0x98) {

        enter_CFI_mode:

            /* Enter CFI query mode */

            pfl->wcycle = 7;

            pfl->cmd = 0x98;

            return;

        }

        if (boff != 0x555 || cmd != 0xAA) {

            DPRINTF("%s: unlock0 failed " TARGET_FMT_lx " %02x %04x\n",

                    __func__, boff, cmd, 0x555);

            goto reset_flash;

        }

        DPRINTF("%s: unlock sequence started\n", __func__);

        break;

    case 1:

        /* We started an unlock sequence */

    check_unlock1:

        if (boff != 0x2AA || cmd != 0x55) {

            DPRINTF("%s: unlock1 failed " TARGET_FMT_lx " %02x\n", __func__,

                    boff, cmd);

            goto reset_flash;

        }

        DPRINTF("%s: unlock sequence done\n", __func__);

        break;

    case 2:

        /* We finished an unlock sequence */

        if (!pfl->bypass && boff != 0x555) {

            DPRINTF("%s: command failed " TARGET_FMT_lx " %02x\n", __func__,

                    boff, cmd);

            goto reset_flash;

        }

        switch (cmd) {

        case 0x20:

            pfl->bypass = 1;

            goto do_bypass;

        case 0x80:

        case 0x90:

        case 0xA0:

            pfl->cmd = cmd;

            DPRINTF("%s: starting command %02x\n", __func__, cmd);

            break;

        default:

            DPRINTF("%s: unknown command %02x\n", __func__, cmd);

            goto reset_flash;

        }

        break;

    case 3:

        switch (pfl->cmd) {

        case 0x80:

            /* We need another unlock sequence */

            goto check_unlock0;

        case 0xA0:

            DPRINTF("%s: write data offset " TARGET_FMT_lx " %08x %d\n",

                    __func__, offset, value, width);

            p = pfl->storage;

            switch (width) {

            case 1:

                p[offset] &= value;

                pflash_update(pfl, offset, 1);

                break;

            case 2:

#if defined(TARGET_WORDS_BIGENDIAN)

                p[offset] &= value >> 8;

                p[offset + 1] &= value;

#else

                p[offset] &= value;

                p[offset + 1] &= value >> 8;

#endif

                pflash_update(pfl, offset, 2);

                break;

            case 4:

#if defined(TARGET_WORDS_BIGENDIAN)

                p[offset] &= value >> 24;

                p[offset + 1] &= value >> 16;

                p[offset + 2] &= value >> 8;

                p[offset + 3] &= value;

#else

                p[offset] &= value;

                p[offset + 1] &= value >> 8;

                p[offset + 2] &= value >> 16;

                p[offset + 3] &= value >> 24;

#endif

                pflash_update(pfl, offset, 4);

                break;

            }

            pfl->status = 0x00 | ~(value & 0x80);

            /* Let's pretend write is immediate */

            if (pfl->bypass)

                goto do_bypass;

            goto reset_flash;

        case 0x90:

            if (pfl->bypass && cmd == 0x00) {

                /* Unlock bypass reset */

                goto reset_flash;

            }

            /* We can enter CFI query mode from autoselect mode */

            if (boff == 0x55 && cmd == 0x98)

                goto enter_CFI_mode;

            /* No break here */

        default:

            DPRINTF("%s: invalid write for command %02x\n",

                    __func__, pfl->cmd);

            goto reset_flash;

        }

    case 4:

        switch (pfl->cmd) {

        case 0xA0:

            /* Ignore writes while flash data write is occuring */

            /* As we suppose write is immediate, this should never happen */

            return;

        case 0x80:

            goto check_unlock1;

        default:

            /* Should never happen */

            DPRINTF("%s: invalid command state %02x (wc 4)\n",

                    __func__, pfl->cmd);

            goto reset_flash;

        }

        break;

    case 5:

        switch (cmd) {

        case 0x10:

            if (boff != 0x555) {

                DPRINTF("%s: chip erase: invalid address " TARGET_FMT_lx "\n",

                        __func__, offset);

                goto reset_flash;

            }

            /* Chip erase */

            DPRINTF("%s: start chip erase\n", __func__);

            memset(pfl->storage, 0xFF, pfl->total_len);

            pfl->status = 0x00;

            pflash_update(pfl, 0, pfl->total_len);

            /* Let's wait 5 seconds before chip erase is done */

            qemu_mod_timer(pfl->timer, 

                           qemu_get_clock(vm_clock) + (ticks_per_sec * 5));

            break;

        case 0x30:

            /* Sector erase */

            p = pfl->storage;

            offset &= ~(pfl->sector_len - 1);

            DPRINTF("%s: start sector erase at " TARGET_FMT_lx "\n", __func__,

                    offset);

            memset(p + offset, 0xFF, pfl->sector_len);

            pflash_update(pfl, offset, pfl->sector_len);

            pfl->status = 0x00;

            /* Let's wait 1/2 second before sector erase is done */

            qemu_mod_timer(pfl->timer, 

                           qemu_get_clock(vm_clock) + (ticks_per_sec / 2));

            break;

        default:

            DPRINTF("%s: invalid command %02x (wc 5)\n", __func__, cmd);

            goto reset_flash;

        }

        pfl->cmd = cmd;

        break;

    case 6:

        switch (pfl->cmd) {

        case 0x10:

            /* Ignore writes during chip erase */

            return;

        case 0x30:

            /* Ignore writes during sector erase */

            return;

        default:

            /* Should never happen */

            DPRINTF("%s: invalid command state %02x (wc 6)\n",

                    __func__, pfl->cmd);

            goto reset_flash;

        }

        break;

    case 7: /* Special value for CFI queries */

        DPRINTF("%s: invalid write in CFI query mode\n", __func__);

        goto reset_flash;

    default:

        /* Should never happen */

        DPRINTF("%s: invalid write state (wc 7)\n",  __func__);

        goto reset_flash;

    }

    pfl->wcycle++;



    return;



    /* Reset flash */

 reset_flash:

    if (pfl->wcycle != 0) {

        cpu_register_physical_memory(pfl->base, pfl->total_len,

                                     pfl->off | IO_MEM_ROMD | pfl->fl_mem);

    }

    pfl->bypass = 0;

    pfl->wcycle = 0;

    pfl->cmd = 0;

    return;



 do_bypass:

    pfl->wcycle = 2;

    pfl->cmd = 0;

    return;

}
