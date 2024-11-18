uint64_t helper_ld_asi(CPUSPARCState *env, target_ulong addr, int asi, int size,

                       int sign)

{

    uint64_t ret = 0;

#if defined(DEBUG_ASI)

    target_ulong last_addr = addr;

#endif



    if (asi < 0x80) {

        helper_raise_exception(env, TT_PRIV_ACT);

    }



    helper_check_align(env, addr, size - 1);

    addr = asi_address_mask(env, asi, addr);



    switch (asi) {

    case 0x82: /* Primary no-fault */

    case 0x8a: /* Primary no-fault LE */

        if (page_check_range(addr, size, PAGE_READ) == -1) {

#ifdef DEBUG_ASI

            dump_asi("read ", last_addr, asi, size, ret);

#endif

            return 0;

        }

        /* Fall through */

    case 0x80: /* Primary */

    case 0x88: /* Primary LE */

        {

            switch (size) {

            case 1:

                ret = ldub_raw(addr);

                break;

            case 2:

                ret = lduw_raw(addr);

                break;

            case 4:

                ret = ldl_raw(addr);

                break;

            default:

            case 8:

                ret = ldq_raw(addr);

                break;

            }

        }

        break;

    case 0x83: /* Secondary no-fault */

    case 0x8b: /* Secondary no-fault LE */

        if (page_check_range(addr, size, PAGE_READ) == -1) {

#ifdef DEBUG_ASI

            dump_asi("read ", last_addr, asi, size, ret);

#endif

            return 0;

        }

        /* Fall through */

    case 0x81: /* Secondary */

    case 0x89: /* Secondary LE */

        /* XXX */

        break;

    default:

        break;

    }



    /* Convert from little endian */

    switch (asi) {

    case 0x88: /* Primary LE */

    case 0x89: /* Secondary LE */

    case 0x8a: /* Primary no-fault LE */

    case 0x8b: /* Secondary no-fault LE */

        switch (size) {

        case 2:

            ret = bswap16(ret);

            break;

        case 4:

            ret = bswap32(ret);

            break;

        case 8:

            ret = bswap64(ret);

            break;

        default:

            break;

        }

    default:

        break;

    }



    /* Convert to signed number */

    if (sign) {

        switch (size) {

        case 1:

            ret = (int8_t) ret;

            break;

        case 2:

            ret = (int16_t) ret;

            break;

        case 4:

            ret = (int32_t) ret;

            break;

        default:

            break;

        }

    }

#ifdef DEBUG_ASI

    dump_asi("read ", last_addr, asi, size, ret);

#endif

    return ret;

}
