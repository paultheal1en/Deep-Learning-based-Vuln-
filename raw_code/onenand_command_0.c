static void onenand_command(OneNANDState *s)

{

    int b;

    int sec;

    void *buf;

#define SETADDR(block, page)			\

    sec = (s->addr[page] & 3) +			\

            ((((s->addr[page] >> 2) & 0x3f) +	\

              (((s->addr[block] & 0xfff) |	\

                (s->addr[block] >> 15 ?		\

                 s->density_mask : 0)) << 6)) << (PAGE_SHIFT - 9));

#define SETBUF_M()				\

    buf = (s->bufaddr & 8) ?			\

            s->data[(s->bufaddr >> 2) & 1][0] : s->boot[0];	\

    buf += (s->bufaddr & 3) << 9;

#define SETBUF_S()				\

    buf = (s->bufaddr & 8) ?			\

            s->data[(s->bufaddr >> 2) & 1][1] : s->boot[1];	\

    buf += (s->bufaddr & 3) << 4;



    switch (s->command) {

    case 0x00:	/* Load single/multiple sector data unit into buffer */

        SETADDR(ONEN_BUF_BLOCK, ONEN_BUF_PAGE)



        SETBUF_M()

        if (onenand_load_main(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_LOAD;



#if 0

        SETBUF_S()

        if (onenand_load_spare(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_LOAD;

#endif



        /* TODO: if (s->bufaddr & 3) + s->count was > 4 (2k-pages)

         * or    if (s->bufaddr & 1) + s->count was > 2 (1k-pages)

         * then we need two split the read/write into two chunks.

         */

        s->intstatus |= ONEN_INT | ONEN_INT_LOAD;

        break;

    case 0x13:	/* Load single/multiple spare sector into buffer */

        SETADDR(ONEN_BUF_BLOCK, ONEN_BUF_PAGE)



        SETBUF_S()

        if (onenand_load_spare(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_LOAD;



        /* TODO: if (s->bufaddr & 3) + s->count was > 4 (2k-pages)

         * or    if (s->bufaddr & 1) + s->count was > 2 (1k-pages)

         * then we need two split the read/write into two chunks.

         */

        s->intstatus |= ONEN_INT | ONEN_INT_LOAD;

        break;

    case 0x80:	/* Program single/multiple sector data unit from buffer */

        SETADDR(ONEN_BUF_BLOCK, ONEN_BUF_PAGE)



        SETBUF_M()

        if (onenand_prog_main(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_PROG;



#if 0

        SETBUF_S()

        if (onenand_prog_spare(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_PROG;

#endif



        /* TODO: if (s->bufaddr & 3) + s->count was > 4 (2k-pages)

         * or    if (s->bufaddr & 1) + s->count was > 2 (1k-pages)

         * then we need two split the read/write into two chunks.

         */

        s->intstatus |= ONEN_INT | ONEN_INT_PROG;

        break;

    case 0x1a:	/* Program single/multiple spare area sector from buffer */

        SETADDR(ONEN_BUF_BLOCK, ONEN_BUF_PAGE)



        SETBUF_S()

        if (onenand_prog_spare(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_PROG;



        /* TODO: if (s->bufaddr & 3) + s->count was > 4 (2k-pages)

         * or    if (s->bufaddr & 1) + s->count was > 2 (1k-pages)

         * then we need two split the read/write into two chunks.

         */

        s->intstatus |= ONEN_INT | ONEN_INT_PROG;

        break;

    case 0x1b:	/* Copy-back program */

        SETBUF_S()



        SETADDR(ONEN_BUF_BLOCK, ONEN_BUF_PAGE)

        if (onenand_load_main(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_PROG;



        SETADDR(ONEN_BUF_DEST_BLOCK, ONEN_BUF_DEST_PAGE)

        if (onenand_prog_main(s, sec, s->count, buf))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_PROG;



        /* TODO: spare areas */



        s->intstatus |= ONEN_INT | ONEN_INT_PROG;

        break;



    case 0x23:	/* Unlock NAND array block(s) */

        s->intstatus |= ONEN_INT;



        /* XXX the previous (?) area should be locked automatically */

        for (b = s->unladdr[0]; b <= s->unladdr[1]; b ++) {

            if (b >= s->blocks) {

                s->status |= ONEN_ERR_CMD;

                break;

            }

            if (s->blockwp[b] == ONEN_LOCK_LOCKTIGHTEN)

                break;



            s->wpstatus = s->blockwp[b] = ONEN_LOCK_UNLOCKED;

        }

        break;

    case 0x27:	/* Unlock All NAND array blocks */

        s->intstatus |= ONEN_INT;



        for (b = 0; b < s->blocks; b ++) {

            if (b >= s->blocks) {

                s->status |= ONEN_ERR_CMD;

                break;

            }

            if (s->blockwp[b] == ONEN_LOCK_LOCKTIGHTEN)

                break;



            s->wpstatus = s->blockwp[b] = ONEN_LOCK_UNLOCKED;

        }

        break;



    case 0x2a:	/* Lock NAND array block(s) */

        s->intstatus |= ONEN_INT;



        for (b = s->unladdr[0]; b <= s->unladdr[1]; b ++) {

            if (b >= s->blocks) {

                s->status |= ONEN_ERR_CMD;

                break;

            }

            if (s->blockwp[b] == ONEN_LOCK_LOCKTIGHTEN)

                break;



            s->wpstatus = s->blockwp[b] = ONEN_LOCK_LOCKED;

        }

        break;

    case 0x2c:	/* Lock-tight NAND array block(s) */

        s->intstatus |= ONEN_INT;



        for (b = s->unladdr[0]; b <= s->unladdr[1]; b ++) {

            if (b >= s->blocks) {

                s->status |= ONEN_ERR_CMD;

                break;

            }

            if (s->blockwp[b] == ONEN_LOCK_UNLOCKED)

                continue;



            s->wpstatus = s->blockwp[b] = ONEN_LOCK_LOCKTIGHTEN;

        }

        break;



    case 0x71:	/* Erase-Verify-Read */

        s->intstatus |= ONEN_INT;

        break;

    case 0x95:	/* Multi-block erase */

        qemu_irq_pulse(s->intr);

        /* Fall through.  */

    case 0x94:	/* Block erase */

        sec = ((s->addr[ONEN_BUF_BLOCK] & 0xfff) |

                        (s->addr[ONEN_BUF_BLOCK] >> 15 ? s->density_mask : 0))

                << (BLOCK_SHIFT - 9);

        if (onenand_erase(s, sec, 1 << (BLOCK_SHIFT - 9)))

            s->status |= ONEN_ERR_CMD | ONEN_ERR_ERASE;



        s->intstatus |= ONEN_INT | ONEN_INT_ERASE;

        break;

    case 0xb0:	/* Erase suspend */

        break;

    case 0x30:	/* Erase resume */

        s->intstatus |= ONEN_INT | ONEN_INT_ERASE;

        break;



    case 0xf0:	/* Reset NAND Flash core */

        onenand_reset(s, 0);

        break;

    case 0xf3:	/* Reset OneNAND */

        onenand_reset(s, 0);

        break;



    case 0x65:	/* OTP Access */

        s->intstatus |= ONEN_INT;

        s->bdrv_cur = NULL;

        s->current = s->otp;

        s->secs_cur = 1 << (BLOCK_SHIFT - 9);

        s->addr[ONEN_BUF_BLOCK] = 0;

        s->otpmode = 1;

        break;



    default:

        s->status |= ONEN_ERR_CMD;

        s->intstatus |= ONEN_INT;

        fprintf(stderr, "%s: unknown OneNAND command %x\n",

                        __func__, s->command);

    }



    onenand_intr_update(s);

}
