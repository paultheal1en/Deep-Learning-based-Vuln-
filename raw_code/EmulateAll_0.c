unsigned int EmulateAll(unsigned int opcode, FPA11* qfpa, CPUARMState* qregs)

{

  unsigned int nRc = 0;

//  unsigned long flags;

  FPA11 *fpa11;

//  save_flags(flags); sti();



  qemufpa=qfpa;

  user_registers=qregs;



#if 0

  fprintf(stderr,"emulating FP insn 0x%08x, PC=0x%08x\n",

          opcode, qregs[REG_PC]);

#endif

  fpa11 = GET_FPA11();



  if (fpa11->initflag == 0)		/* good place for __builtin_expect */

  {

    resetFPA11();

    SetRoundingMode(ROUND_TO_NEAREST);

    SetRoundingPrecision(ROUND_EXTENDED);

    fpa11->initflag = 1;

  }



  set_float_exception_flags(0, &fpa11->fp_status);



  if (TEST_OPCODE(opcode,MASK_CPRT))

  {

    //fprintf(stderr,"emulating CPRT\n");

    /* Emulate conversion opcodes. */

    /* Emulate register transfer opcodes. */

    /* Emulate comparison opcodes. */

    nRc = EmulateCPRT(opcode);

  }

  else if (TEST_OPCODE(opcode,MASK_CPDO))

  {

    //fprintf(stderr,"emulating CPDO\n");

    /* Emulate monadic arithmetic opcodes. */

    /* Emulate dyadic arithmetic opcodes. */

    nRc = EmulateCPDO(opcode);

  }

  else if (TEST_OPCODE(opcode,MASK_CPDT))

  {

    //fprintf(stderr,"emulating CPDT\n");

    /* Emulate load/store opcodes. */

    /* Emulate load/store multiple opcodes. */

    nRc = EmulateCPDT(opcode);

  }

  else

  {

    /* Invalid instruction detected.  Return FALSE. */

    nRc = 0;

  }



//  restore_flags(flags);

  if(nRc == 1 && get_float_exception_flags(&fpa11->fp_status))

  {

    //printf("fef 0x%x\n",float_exception_flags);

    nRc -= get_float_exception_flags(&fpa11->fp_status);

  }



  //printf("returning %d\n",nRc);

  return(nRc);

}
