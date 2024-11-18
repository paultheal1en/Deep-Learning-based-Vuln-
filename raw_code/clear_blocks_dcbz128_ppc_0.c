void clear_blocks_dcbz128_ppc(DCTELEM *blocks)

{

POWERPC_TBL_DECLARE(powerpc_clear_blocks_dcbz128, 1);

    register int misal = ((unsigned long)blocks & 0x0000007f);

    register int i = 0;

POWERPC_TBL_START_COUNT(powerpc_clear_blocks_dcbz128, 1);

#if 1

 if (misal) {

   // we could probably also optimize this case,

   // but there's not much point as the machines

   // aren't available yet (2003-06-26)

      memset(blocks, 0, sizeof(DCTELEM)*6*64);

    }

    else

      for ( ; i < sizeof(DCTELEM)*6*64 ; i += 128) {

	asm volatile("dcbzl %0,%1" : : "b" (blocks), "r" (i) : "memory");

      }

#else

    memset(blocks, 0, sizeof(DCTELEM)*6*64);

#endif

POWERPC_TBL_STOP_COUNT(powerpc_clear_blocks_dcbz128, 1);

}
