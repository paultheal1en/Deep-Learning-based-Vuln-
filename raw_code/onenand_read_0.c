static uint64_t onenand_read(void *opaque, hwaddr addr,

                             unsigned size)

{

    OneNANDState *s = (OneNANDState *) opaque;

    int offset = addr >> s->shift;



    switch (offset) {

    case 0x0000 ... 0xc000:

        return lduw_le_p(s->boot[0] + addr);



    case 0xf000:	/* Manufacturer ID */

        return s->id.man;

    case 0xf001:	/* Device ID */

        return s->id.dev;

    case 0xf002:	/* Version ID */

        return s->id.ver;

    /* TODO: get the following values from a real chip!  */

    case 0xf003:	/* Data Buffer size */

        return 1 << PAGE_SHIFT;

    case 0xf004:	/* Boot Buffer size */

        return 0x200;

    case 0xf005:	/* Amount of buffers */

        return 1 | (2 << 8);

    case 0xf006:	/* Technology */

        return 0;



    case 0xf100 ... 0xf107:	/* Start addresses */

        return s->addr[offset - 0xf100];



    case 0xf200:	/* Start buffer */

        return (s->bufaddr << 8) | ((s->count - 1) & (1 << (PAGE_SHIFT - 10)));



    case 0xf220:	/* Command */

        return s->command;

    case 0xf221:	/* System Configuration 1 */

        return s->config[0] & 0xffe0;

    case 0xf222:	/* System Configuration 2 */

        return s->config[1];



    case 0xf240:	/* Controller Status */

        return s->status;

    case 0xf241:	/* Interrupt */

        return s->intstatus;

    case 0xf24c:	/* Unlock Start Block Address */

        return s->unladdr[0];

    case 0xf24d:	/* Unlock End Block Address */

        return s->unladdr[1];

    case 0xf24e:	/* Write Protection Status */

        return s->wpstatus;



    case 0xff00:	/* ECC Status */

        return 0x00;

    case 0xff01:	/* ECC Result of main area data */

    case 0xff02:	/* ECC Result of spare area data */

    case 0xff03:	/* ECC Result of main area data */

    case 0xff04:	/* ECC Result of spare area data */

        hw_error("%s: imeplement ECC\n", __FUNCTION__);

        return 0x0000;

    }



    fprintf(stderr, "%s: unknown OneNAND register %x\n",

                    __FUNCTION__, offset);

    return 0;

}
