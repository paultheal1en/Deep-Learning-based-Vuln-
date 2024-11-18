print_insn_sparc (bfd_vma memaddr, disassemble_info *info)

{

  FILE *stream = info->stream;

  bfd_byte buffer[4];

  unsigned long insn;

  sparc_opcode_hash *op;

  /* Nonzero of opcode table has been initialized.  */

  static int opcodes_initialized = 0;

  /* bfd mach number of last call.  */

  static unsigned long current_mach = 0;

  bfd_vma (*getword) (const unsigned char *);



  if (!opcodes_initialized

      || info->mach != current_mach)

    {

      int i;



      current_arch_mask = compute_arch_mask (info->mach);



      if (!opcodes_initialized)

        sorted_opcodes =

          malloc (sparc_num_opcodes * sizeof (sparc_opcode *));

      /* Reset the sorted table so we can resort it.  */

      for (i = 0; i < sparc_num_opcodes; ++i)

        sorted_opcodes[i] = &sparc_opcodes[i];

      qsort ((char *) sorted_opcodes, sparc_num_opcodes,

             sizeof (sorted_opcodes[0]), compare_opcodes);



      build_hash_table (sorted_opcodes, opcode_hash_table, sparc_num_opcodes);

      current_mach = info->mach;

      opcodes_initialized = 1;

    }



  {

    int status =

      (*info->read_memory_func) (memaddr, buffer, sizeof (buffer), info);



    if (status != 0)

      {

        (*info->memory_error_func) (status, memaddr, info);

        return -1;

      }

  }



  /* On SPARClite variants such as DANlite (sparc86x), instructions

     are always big-endian even when the machine is in little-endian mode.  */

  if (info->endian == BFD_ENDIAN_BIG || info->mach == bfd_mach_sparc_sparclite)

    getword = bfd_getb32;

  else

    getword = bfd_getl32;



  insn = getword (buffer);



  info->insn_info_valid = 1;                    /* We do return this info.  */

  info->insn_type = dis_nonbranch;              /* Assume non branch insn.  */

  info->branch_delay_insns = 0;                 /* Assume no delay.  */

  info->target = 0;                             /* Assume no target known.  */



  for (op = opcode_hash_table[HASH_INSN (insn)]; op; op = op->next)

    {

      const sparc_opcode *opcode = op->opcode;



      /* If the insn isn't supported by the current architecture, skip it.  */

      if (! (opcode->architecture & current_arch_mask))

        continue;



      if ((opcode->match & insn) == opcode->match

          && (opcode->lose & insn) == 0)

        {

          /* Nonzero means that we have found an instruction which has

             the effect of adding or or'ing the imm13 field to rs1.  */

          int imm_added_to_rs1 = 0;

          int imm_ored_to_rs1 = 0;



          /* Nonzero means that we have found a plus sign in the args

             field of the opcode table.  */

          int found_plus = 0;



          /* Nonzero means we have an annulled branch.  */

          /* int is_annulled = 0; */ /* see FIXME below */



          /* Do we have an `add' or `or' instruction combining an

             immediate with rs1?  */

          if (opcode->match == 0x80102000) /* or */

            imm_ored_to_rs1 = 1;

          if (opcode->match == 0x80002000) /* add */

            imm_added_to_rs1 = 1;



          if (X_RS1 (insn) != X_RD (insn)

              && strchr (opcode->args, 'r') != NULL)

              /* Can't do simple format if source and dest are different.  */

              continue;

          if (X_RS2 (insn) != X_RD (insn)

              && strchr (opcode->args, 'O') != NULL)

              /* Can't do simple format if source and dest are different.  */

              continue;



          (*info->fprintf_func) (stream, "%s", opcode->name);



          {

            const char *s;



            if (opcode->args[0] != ',')

              (*info->fprintf_func) (stream, " ");



            for (s = opcode->args; *s != '\0'; ++s)

              {

                while (*s == ',')

                  {

                    (*info->fprintf_func) (stream, ",");

                    ++s;

                    switch (*s)

                      {

                      case 'a':

                        (*info->fprintf_func) (stream, "a");

                        /* is_annulled = 1; */ /* see FIXME below */

                        ++s;

                        continue;

                      case 'N':

                        (*info->fprintf_func) (stream, "pn");

                        ++s;

                        continue;



                      case 'T':

                        (*info->fprintf_func) (stream, "pt");

                        ++s;

                        continue;



                      default:

                        break;

                      }

                  }



                (*info->fprintf_func) (stream, " ");



                switch (*s)

                  {

                  case '+':

                    found_plus = 1;

                    /* Fall through.  */



                  default:

                    (*info->fprintf_func) (stream, "%c", *s);

                    break;



                  case '#':

                    (*info->fprintf_func) (stream, "0");

                    break;



#define reg(n)  (*info->fprintf_func) (stream, "%%%s", reg_names[n])

                  case '1':

                  case 'r':

                    reg (X_RS1 (insn));

                    break;



                  case '2':

                  case 'O':

                    reg (X_RS2 (insn));

                    break;



                  case 'd':

                    reg (X_RD (insn));

                    break;

#undef  reg



#define freg(n)         (*info->fprintf_func) (stream, "%%%s", freg_names[n])

#define fregx(n)        (*info->fprintf_func) (stream, "%%%s", freg_names[((n) & ~1) | (((n) & 1) << 5)])

                  case 'e':

                    freg (X_RS1 (insn));

                    break;

                  case 'v':     /* Double/even.  */

                  case 'V':     /* Quad/multiple of 4.  */

                    fregx (X_RS1 (insn));

                    break;



                  case 'f':

                    freg (X_RS2 (insn));

                    break;

                  case 'B':     /* Double/even.  */

                  case 'R':     /* Quad/multiple of 4.  */

                    fregx (X_RS2 (insn));

                    break;



                  case 'g':

                    freg (X_RD (insn));

                    break;

                  case 'H':     /* Double/even.  */

                  case 'J':     /* Quad/multiple of 4.  */

                    fregx (X_RD (insn));

                    break;

#undef  freg

#undef  fregx



#define creg(n) (*info->fprintf_func) (stream, "%%c%u", (unsigned int) (n))

                  case 'b':

                    creg (X_RS1 (insn));

                    break;



                  case 'c':

                    creg (X_RS2 (insn));

                    break;



                  case 'D':

                    creg (X_RD (insn));

                    break;

#undef  creg



                  case 'h':

                    (*info->fprintf_func) (stream, "%%hi(%#x)",

                                           ((unsigned) 0xFFFFFFFF

                                            & ((int) X_IMM22 (insn) << 10)));

                    break;



                  case 'i':     /* 13 bit immediate.  */

                  case 'I':     /* 11 bit immediate.  */

                  case 'j':     /* 10 bit immediate.  */

                    {

                      int imm;



                      if (*s == 'i')

                        imm = X_SIMM (insn, 13);

                      else if (*s == 'I')

                        imm = X_SIMM (insn, 11);

                      else

                        imm = X_SIMM (insn, 10);



                      /* Check to see whether we have a 1+i, and take

                         note of that fact.



                         Note: because of the way we sort the table,

                         we will be matching 1+i rather than i+1,

                         so it is OK to assume that i is after +,

                         not before it.  */

                      if (found_plus)

                        imm_added_to_rs1 = 1;



                      if (imm <= 9)

                        (*info->fprintf_func) (stream, "%d", imm);

                      else

                        (*info->fprintf_func) (stream, "%#x", imm);

                    }

                    break;



                  case 'X':     /* 5 bit unsigned immediate.  */

                  case 'Y':     /* 6 bit unsigned immediate.  */

                    {

                      int imm = X_IMM (insn, *s == 'X' ? 5 : 6);



                      if (imm <= 9)

                        (info->fprintf_func) (stream, "%d", imm);

                      else

                        (info->fprintf_func) (stream, "%#x", (unsigned) imm);

                    }

                    break;



                  case '3':

                    (info->fprintf_func) (stream, "%ld", X_IMM (insn, 3));

                    break;



                  case 'K':

                    {

                      int mask = X_MEMBAR (insn);

                      int bit = 0x40, printed_one = 0;

                      const char *name;



                      if (mask == 0)

                        (info->fprintf_func) (stream, "0");

                      else

                        while (bit)

                          {

                            if (mask & bit)

                              {

                                if (printed_one)

                                  (info->fprintf_func) (stream, "|");

                                name = sparc_decode_membar (bit);

                                (info->fprintf_func) (stream, "%s", name);

                                printed_one = 1;

                              }

                            bit >>= 1;

                          }

                      break;

                    }



                  case 'k':

                    info->target = memaddr + SEX (X_DISP16 (insn), 16) * 4;

                    (*info->print_address_func) (info->target, info);

                    break;



                  case 'G':

                    info->target = memaddr + SEX (X_DISP19 (insn), 19) * 4;

                    (*info->print_address_func) (info->target, info);

                    break;



                  case '6':

                  case '7':

                  case '8':

                  case '9':

                    (*info->fprintf_func) (stream, "%%fcc%c", *s - '6' + '0');

                    break;



                  case 'z':

                    (*info->fprintf_func) (stream, "%%icc");

                    break;



                  case 'Z':

                    (*info->fprintf_func) (stream, "%%xcc");

                    break;



                  case 'E':

                    (*info->fprintf_func) (stream, "%%ccr");

                    break;



                  case 's':

                    (*info->fprintf_func) (stream, "%%fprs");

                    break;



                  case 'o':

                    (*info->fprintf_func) (stream, "%%asi");

                    break;



                  case 'W':

                    (*info->fprintf_func) (stream, "%%tick");

                    break;



                  case 'P':

                    (*info->fprintf_func) (stream, "%%pc");

                    break;



                  case '?':

                    if (X_RS1 (insn) == 31)

                      (*info->fprintf_func) (stream, "%%ver");

                    else if ((unsigned) X_RS1 (insn) < 17)

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9_priv_reg_names[X_RS1 (insn)]);

                    else

                      (*info->fprintf_func) (stream, "%%reserved");

                    break;



                  case '!':

                    if ((unsigned) X_RD (insn) < 17)

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9_priv_reg_names[X_RD (insn)]);

                    else

                      (*info->fprintf_func) (stream, "%%reserved");

                    break;



                  case '$':

                    if ((unsigned) X_RS1 (insn) < 32)

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9_hpriv_reg_names[X_RS1 (insn)]);

                    else

                      (*info->fprintf_func) (stream, "%%reserved");

                    break;



                  case '%':

                    if ((unsigned) X_RD (insn) < 32)

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9_hpriv_reg_names[X_RD (insn)]);

                    else

                      (*info->fprintf_func) (stream, "%%reserved");

                    break;



                  case '/':

                    if (X_RS1 (insn) < 16 || X_RS1 (insn) > 25)

                      (*info->fprintf_func) (stream, "%%reserved");

                    else

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9a_asr_reg_names[X_RS1 (insn)-16]);

                    break;



                  case '_':

                    if (X_RD (insn) < 16 || X_RD (insn) > 25)

                      (*info->fprintf_func) (stream, "%%reserved");

                    else

                      (*info->fprintf_func) (stream, "%%%s",

                                             v9a_asr_reg_names[X_RD (insn)-16]);

                    break;



                  case '*':

                    {

                      const char *name = sparc_decode_prefetch (X_RD (insn));



                      if (name)

                        (*info->fprintf_func) (stream, "%s", name);

                      else

                        (*info->fprintf_func) (stream, "%ld", X_RD (insn));

                      break;

                    }



                  case 'M':

                    (*info->fprintf_func) (stream, "%%asr%ld", X_RS1 (insn));

                    break;



                  case 'm':

                    (*info->fprintf_func) (stream, "%%asr%ld", X_RD (insn));

                    break;



                  case 'L':

                    info->target = memaddr + SEX (X_DISP30 (insn), 30) * 4;

                    (*info->print_address_func) (info->target, info);

                    break;



                  case 'n':

                    (*info->fprintf_func)

                      (stream, "%#x", SEX (X_DISP22 (insn), 22));

                    break;



                  case 'l':

                    info->target = memaddr + SEX (X_DISP22 (insn), 22) * 4;

                    (*info->print_address_func) (info->target, info);

                    break;



                  case 'A':

                    {

                      const char *name;



                      if ((info->mach == bfd_mach_sparc_v8plusa) ||

                          ((info->mach >= bfd_mach_sparc_v9) &&

                           (info->mach <= bfd_mach_sparc_v9b)))

                        name = sparc_decode_asi_v9 (X_ASI (insn));

                      else

                        name = sparc_decode_asi_v8 (X_ASI (insn));



                      if (name)

                        (*info->fprintf_func) (stream, "%s", name);

                      else

                        (*info->fprintf_func) (stream, "(%ld)", X_ASI (insn));

                      break;

                    }



                  case 'C':

                    (*info->fprintf_func) (stream, "%%csr");

                    break;



                  case 'F':

                    (*info->fprintf_func) (stream, "%%fsr");

                    break;



                  case 'p':

                    (*info->fprintf_func) (stream, "%%psr");

                    break;



                  case 'q':

                    (*info->fprintf_func) (stream, "%%fq");

                    break;



                  case 'Q':

                    (*info->fprintf_func) (stream, "%%cq");

                    break;



                  case 't':

                    (*info->fprintf_func) (stream, "%%tbr");

                    break;



                  case 'w':

                    (*info->fprintf_func) (stream, "%%wim");

                    break;



                  case 'x':

                    (*info->fprintf_func) (stream, "%ld",

                                           ((X_LDST_I (insn) << 8)

                                            + X_ASI (insn)));

                    break;



                  case 'y':

                    (*info->fprintf_func) (stream, "%%y");

                    break;



                  case 'u':

                  case 'U':

                    {

                      int val = *s == 'U' ? X_RS1 (insn) : X_RD (insn);

                      const char *name = sparc_decode_sparclet_cpreg (val);



                      if (name)

                        (*info->fprintf_func) (stream, "%s", name);

                      else

                        (*info->fprintf_func) (stream, "%%cpreg(%d)", val);

                      break;

                    }

                  }

              }

          }



          /* If we are adding or or'ing something to rs1, then

             check to see whether the previous instruction was

             a sethi to the same register as in the sethi.

             If so, attempt to print the result of the add or

             or (in this context add and or do the same thing)

             and its symbolic value.  */

          if (imm_ored_to_rs1 || imm_added_to_rs1)

            {

              unsigned long prev_insn;

              int errcode;



              if (memaddr >= 4)

                errcode =

                  (*info->read_memory_func)

                  (memaddr - 4, buffer, sizeof (buffer), info);

              else

                errcode = 1;



              prev_insn = getword (buffer);



              if (errcode == 0)

                {

                  /* If it is a delayed branch, we need to look at the

                     instruction before the delayed branch.  This handles

                     sequences such as:



                     sethi %o1, %hi(_foo), %o1

                     call _printf

                     or %o1, %lo(_foo), %o1  */



                  if (is_delayed_branch (prev_insn))

                    {

                      if (memaddr >= 8)

                        errcode = (*info->read_memory_func)

                          (memaddr - 8, buffer, sizeof (buffer), info);

                      else

                        errcode = 1;



                      prev_insn = getword (buffer);

                    }

                }



              /* If there was a problem reading memory, then assume

                 the previous instruction was not sethi.  */

              if (errcode == 0)

                {

                  /* Is it sethi to the same register?  */

                  if ((prev_insn & 0xc1c00000) == 0x01000000

                      && X_RD (prev_insn) == X_RS1 (insn))

                    {

                      (*info->fprintf_func) (stream, "\t! ");

                      info->target =

                        ((unsigned) 0xFFFFFFFF

                         & ((int) X_IMM22 (prev_insn) << 10));

                      if (imm_added_to_rs1)

                        info->target += X_SIMM (insn, 13);

                      else

                        info->target |= X_SIMM (insn, 13);

                      (*info->print_address_func) (info->target, info);

                      info->insn_type = dis_dref;

                      info->data_size = 4;  /* FIXME!!! */

                    }

                }

            }



          if (opcode->flags & (F_UNBR|F_CONDBR|F_JSR))

            {

                /* FIXME -- check is_annulled flag.  */

              if (opcode->flags & F_UNBR)

                info->insn_type = dis_branch;

              if (opcode->flags & F_CONDBR)

                info->insn_type = dis_condbranch;

              if (opcode->flags & F_JSR)

                info->insn_type = dis_jsr;

              if (opcode->flags & F_DELAYED)

                info->branch_delay_insns = 1;

            }



          return sizeof (buffer);

        }

    }



  info->insn_type = dis_noninsn;        /* Mark as non-valid instruction.  */

  (*info->fprintf_func) (stream, _("unknown"));

  return sizeof (buffer);

}
