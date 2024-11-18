print_with_operands (const struct cris_opcode *opcodep,

		     unsigned int insn,

		     unsigned char *buffer,

		     bfd_vma addr,

		     disassemble_info *info,

		     /* If a prefix insn was before this insn (and is supposed

			to be output as an address), here is a description of

			it.  */

		     const struct cris_opcode *prefix_opcodep,

		     unsigned int prefix_insn,

		     unsigned char *prefix_buffer,

		     bfd_boolean with_reg_prefix)

{

  /* Get a buffer of somewhat reasonable size where we store

     intermediate parts of the insn.  */

  char temp[sizeof (".d [$r13=$r12-2147483648],$r10") * 2];

  char *tp = temp;

  static const char mode_char[] = "bwd?";

  const char *s;

  const char *cs;

  struct cris_disasm_data *disdata

    = (struct cris_disasm_data *) info->private_data;



  /* Print out the name first thing we do.  */

  (*info->fprintf_func) (info->stream, "%s", opcodep->name);



  cs = opcodep->args;

  s = cs;



  /* Ignore any prefix indicator.  */

  if (*s == 'p')

    s++;



  if (*s == 'm' || *s == 'M' || *s == 'z')

    {

      *tp++ = '.';



      /* Get the size-letter.  */

      *tp++ = *s == 'M'

	? (insn & 0x8000 ? 'd'

	   : insn & 0x4000 ? 'w' : 'b')

	: mode_char[(insn >> 4) & (*s == 'z' ? 1 : 3)];



      /* Ignore the size and the space character that follows.  */

      s += 2;

    }



  /* Add a space if this isn't a long-branch, because for those will add

     the condition part of the name later.  */

  if (opcodep->match != (BRANCH_PC_LOW + BRANCH_INCR_HIGH * 256))

    *tp++ = ' ';



  /* Fill in the insn-type if deducible from the name (and there's no

     better way).  */

  if (opcodep->name[0] == 'j')

    {

      if (CONST_STRNEQ (opcodep->name, "jsr"))

	/* It's "jsr" or "jsrc".  */

	info->insn_type = dis_jsr;

      else

	/* Any other jump-type insn is considered a branch.  */

	info->insn_type = dis_branch;

    }



  /* We might know some more fields right now.  */

  info->branch_delay_insns = opcodep->delayed;



  /* Handle operands.  */

  for (; *s; s++)

    {

    switch (*s)

      {

      case 'T':

	tp = format_sup_reg ((insn >> 12) & 15, tp, with_reg_prefix);

	break;



      case 'A':

	if (with_reg_prefix)

	  *tp++ = REGISTER_PREFIX_CHAR;

	*tp++ = 'a';

	*tp++ = 'c';

	*tp++ = 'r';

	break;



      case '[':

      case ']':

      case ',':

	*tp++ = *s;

	break;



      case '!':

	/* Ignore at this point; used at earlier stages to avoid

	   recognition if there's a prefix at something that in other

	   ways looks like a "pop".  */

	break;



      case 'd':

	/* Ignore.  This is an optional ".d " on the large one of

	   relaxable insns.  */

	break;



      case 'B':

	/* This was the prefix that made this a "push".  We've already

	   handled it by recognizing it, so signal that the prefix is

	   handled by setting it to NULL.  */

	prefix_opcodep = NULL;

	break;



      case 'D':

      case 'r':

	tp = format_reg (disdata, insn & 15, tp, with_reg_prefix);

	break;



      case 'R':

	tp = format_reg (disdata, (insn >> 12) & 15, tp, with_reg_prefix);

	break;



      case 'n':

	{

	  /* Like N but pc-relative to the start of the insn.  */

	  unsigned long number

	    = (buffer[2] + buffer[3] * 256 + buffer[4] * 65536

	       + buffer[5] * 0x1000000 + addr);



	  /* Finish off and output previous formatted bytes.  */

	  *tp = 0;

	  if (temp[0])

	    (*info->fprintf_func) (info->stream, "%s", temp);

	  tp = temp;



	  (*info->print_address_func) ((bfd_vma) number, info);

	}

	break;



      case 'u':

	{

	  /* Like n but the offset is bits <3:0> in the instruction.  */

	  unsigned long number = (buffer[0] & 0xf) * 2 + addr;



	  /* Finish off and output previous formatted bytes.  */

	  *tp = 0;

	  if (temp[0])

	    (*info->fprintf_func) (info->stream, "%s", temp);

	  tp = temp;



	  (*info->print_address_func) ((bfd_vma) number, info);

	}

	break;



      case 'N':

      case 'y':

      case 'Y':

      case 'S':

      case 's':

	/* Any "normal" memory operand.  */

	if ((insn & 0x400) && (insn & 15) == 15 && prefix_opcodep == NULL)

	  {

	    /* We're looking at [pc+], i.e. we need to output an immediate

	       number, where the size can depend on different things.  */

	    long number;

	    int signedp

	      = ((*cs == 'z' && (insn & 0x20))

		 || opcodep->match == BDAP_QUICK_OPCODE);

	    int nbytes;



	    if (opcodep->imm_oprnd_size == SIZE_FIX_32)

	      nbytes = 4;

	    else if (opcodep->imm_oprnd_size == SIZE_SPEC_REG)

	      {

		const struct cris_spec_reg *sregp

		  = spec_reg_info ((insn >> 12) & 15, disdata->distype);



		/* A NULL return should have been as a non-match earlier,

		   so catch it as an internal error in the error-case

		   below.  */

		if (sregp == NULL)

		  /* Whatever non-valid size.  */

		  nbytes = 42;

		else

		  /* PC is always incremented by a multiple of two.

		     For CRISv32, immediates are always 4 bytes for

		     special registers.  */

		  nbytes = disdata->distype == cris_dis_v32

		    ? 4 : (sregp->reg_size + 1) & ~1;

	      }

	    else

	      {

		int mode_size = 1 << ((insn >> 4) & (*cs == 'z' ? 1 : 3));



		if (mode_size == 1)

		  nbytes = 2;

		else

		  nbytes = mode_size;

	      }



	    switch (nbytes)

	      {

	      case 1:

		number = buffer[2];

		if (signedp && number > 127)

		  number -= 256;

		break;



	      case 2:

		number = buffer[2] + buffer[3] * 256;

		if (signedp && number > 32767)

		  number -= 65536;

		break;



	      case 4:

		number

		  = buffer[2] + buffer[3] * 256 + buffer[4] * 65536

		  + buffer[5] * 0x1000000;

		break;



	      default:

		strcpy (tp, "bug");

		tp += 3;

		number = 42;

	      }



	    if ((*cs == 'z' && (insn & 0x20))

		|| (opcodep->match == BDAP_QUICK_OPCODE

		    && (nbytes <= 2 || buffer[1 + nbytes] == 0)))

	      tp = format_dec (number, tp, signedp);

	    else

	      {

		unsigned int highbyte = (number >> 24) & 0xff;



		/* Either output this as an address or as a number.  If it's

		   a dword with the same high-byte as the address of the

		   insn, assume it's an address, and also if it's a non-zero

		   non-0xff high-byte.  If this is a jsr or a jump, then

		   it's definitely an address.  */

		if (nbytes == 4

		    && (highbyte == ((addr >> 24) & 0xff)

			|| (highbyte != 0 && highbyte != 0xff)

			|| info->insn_type == dis_branch

			|| info->insn_type == dis_jsr))

		  {

		    /* Finish off and output previous formatted bytes.  */

		    *tp = 0;

		    tp = temp;

		    if (temp[0])

		      (*info->fprintf_func) (info->stream, "%s", temp);



		    (*info->print_address_func) ((bfd_vma) number, info);



		    info->target = number;

		  }

		else

		  tp = format_hex (number, tp, disdata);

	      }

	  }

	else

	  {

	    /* Not an immediate number.  Then this is a (possibly

	       prefixed) memory operand.  */

	    if (info->insn_type != dis_nonbranch)

	      {

		int mode_size

		  = 1 << ((insn >> 4)

			  & (opcodep->args[0] == 'z' ? 1 : 3));

		int size;

		info->insn_type = dis_dref;

		info->flags |= CRIS_DIS_FLAG_MEMREF;



		if (opcodep->imm_oprnd_size == SIZE_FIX_32)

		  size = 4;

		else if (opcodep->imm_oprnd_size == SIZE_SPEC_REG)

		  {

		    const struct cris_spec_reg *sregp

		      = spec_reg_info ((insn >> 12) & 15, disdata->distype);



		    /* FIXME: Improve error handling; should have been caught

		       earlier.  */

		    if (sregp == NULL)

		      size = 4;

		    else

		      size = sregp->reg_size;

		  }

		else

		  size = mode_size;



		info->data_size = size;

	      }



	    *tp++ = '[';



	    if (prefix_opcodep

		/* We don't match dip with a postincremented field

		   as a side-effect address mode.  */

		&& ((insn & 0x400) == 0

		    || prefix_opcodep->match != DIP_OPCODE))

	      {

		if (insn & 0x400)

		  {

		    tp = format_reg (disdata, insn & 15, tp, with_reg_prefix);

		    *tp++ = '=';

		  }





		/* We mainly ignore the prefix format string when the

		   address-mode syntax is output.  */

		switch (prefix_opcodep->match)

		  {

		  case DIP_OPCODE:

		    /* It's [r], [r+] or [pc+].  */

		    if ((prefix_insn & 0x400) && (prefix_insn & 15) == 15)

		      {

			/* It's [pc+].  This cannot possibly be anything

			   but an address.  */

			unsigned long number

			  = prefix_buffer[2] + prefix_buffer[3] * 256

			  + prefix_buffer[4] * 65536

			  + prefix_buffer[5] * 0x1000000;



			info->target = (bfd_vma) number;



			/* Finish off and output previous formatted

			   data.  */

			*tp = 0;

			tp = temp;

			if (temp[0])

			  (*info->fprintf_func) (info->stream, "%s", temp);



			(*info->print_address_func) ((bfd_vma) number, info);

		      }

		    else

		      {

			/* For a memref in an address, we use target2.

			   In this case, target is zero.  */

			info->flags

			  |= (CRIS_DIS_FLAG_MEM_TARGET2_IS_REG

			      | CRIS_DIS_FLAG_MEM_TARGET2_MEM);



			info->target2 = prefix_insn & 15;



			*tp++ = '[';

			tp = format_reg (disdata, prefix_insn & 15, tp,

					 with_reg_prefix);

			if (prefix_insn & 0x400)

			  *tp++ = '+';

			*tp++ = ']';

		      }

		    break;



		  case BDAP_QUICK_OPCODE:

		    {

		      int number;



		      number = prefix_buffer[0];

		      if (number > 127)

			number -= 256;



		      /* Output "reg+num" or, if num < 0, "reg-num".  */

		      tp = format_reg (disdata, (prefix_insn >> 12) & 15, tp,

				       with_reg_prefix);

		      if (number >= 0)

			*tp++ = '+';

		      tp = format_dec (number, tp, 1);



		      info->flags |= CRIS_DIS_FLAG_MEM_TARGET_IS_REG;

		      info->target = (prefix_insn >> 12) & 15;

		      info->target2 = (bfd_vma) number;

		      break;

		    }



		  case BIAP_OPCODE:

		    /* Output "r+R.m".  */

		    tp = format_reg (disdata, prefix_insn & 15, tp,

				     with_reg_prefix);

		    *tp++ = '+';

		    tp = format_reg (disdata, (prefix_insn >> 12) & 15, tp,

				     with_reg_prefix);

		    *tp++ = '.';

		    *tp++ = mode_char[(prefix_insn >> 4) & 3];



		    info->flags

		      |= (CRIS_DIS_FLAG_MEM_TARGET2_IS_REG

			  | CRIS_DIS_FLAG_MEM_TARGET_IS_REG



			  | ((prefix_insn & 0x8000)

			     ? CRIS_DIS_FLAG_MEM_TARGET2_MULT4

			     : ((prefix_insn & 0x8000)

				? CRIS_DIS_FLAG_MEM_TARGET2_MULT2 : 0)));



		    /* Is it the casejump?  It's a "adds.w [pc+r%d.w],pc".  */

		    if (insn == 0xf83f && (prefix_insn & ~0xf000) == 0x55f)

		      /* Then start interpreting data as offsets.  */

		      case_offset_counter = no_of_case_offsets;

		    break;



		  case BDAP_INDIR_OPCODE:

		    /* Output "r+s.m", or, if "s" is [pc+], "r+s" or

		       "r-s".  */

		    tp = format_reg (disdata, (prefix_insn >> 12) & 15, tp,

				     with_reg_prefix);



		    if ((prefix_insn & 0x400) && (prefix_insn & 15) == 15)

		      {

			long number;

			unsigned int nbytes;



			/* It's a value.  Get its size.  */

			int mode_size = 1 << ((prefix_insn >> 4) & 3);



			if (mode_size == 1)

			  nbytes = 2;

			else

			  nbytes = mode_size;



			switch (nbytes)

			  {

			  case 1:

			    number = prefix_buffer[2];

			    if (number > 127)

			      number -= 256;

			    break;



			  case 2:

			    number = prefix_buffer[2] + prefix_buffer[3] * 256;

			    if (number > 32767)

			      number -= 65536;

			    break;



			  case 4:

			    number

			      = prefix_buffer[2] + prefix_buffer[3] * 256

			      + prefix_buffer[4] * 65536

			      + prefix_buffer[5] * 0x1000000;

			    break;



			  default:

			    strcpy (tp, "bug");

			    tp += 3;

			    number = 42;

			  }



			info->flags |= CRIS_DIS_FLAG_MEM_TARGET_IS_REG;

			info->target2 = (bfd_vma) number;



			/* If the size is dword, then assume it's an

			   address.  */

			if (nbytes == 4)

			  {

			    /* Finish off and output previous formatted

			       bytes.  */

			    *tp++ = '+';

			    *tp = 0;

			    tp = temp;

			    (*info->fprintf_func) (info->stream, "%s", temp);



			    (*info->print_address_func) ((bfd_vma) number, info);

			  }

			else

			  {

			    if (number >= 0)

			      *tp++ = '+';

			    tp = format_dec (number, tp, 1);

			  }

		      }

		    else

		      {

			/* Output "r+[R].m" or "r+[R+].m".  */

			*tp++ = '+';

			*tp++ = '[';

			tp = format_reg (disdata, prefix_insn & 15, tp,

					 with_reg_prefix);

			if (prefix_insn & 0x400)

			  *tp++ = '+';

			*tp++ = ']';

			*tp++ = '.';

			*tp++ = mode_char[(prefix_insn >> 4) & 3];



			info->flags

			  |= (CRIS_DIS_FLAG_MEM_TARGET2_IS_REG

			      | CRIS_DIS_FLAG_MEM_TARGET2_MEM

			      | CRIS_DIS_FLAG_MEM_TARGET_IS_REG



			      | (((prefix_insn >> 4) == 2)

				 ? 0

				 : (((prefix_insn >> 4) & 3) == 1

				    ? CRIS_DIS_FLAG_MEM_TARGET2_MEM_WORD

				    : CRIS_DIS_FLAG_MEM_TARGET2_MEM_BYTE)));

		      }

		    break;



		  default:

		    (*info->fprintf_func) (info->stream, "?prefix-bug");

		  }



		/* To mark that the prefix is used, reset it.  */

		prefix_opcodep = NULL;

	      }

	    else

	      {

		tp = format_reg (disdata, insn & 15, tp, with_reg_prefix);



		info->flags |= CRIS_DIS_FLAG_MEM_TARGET_IS_REG;

		info->target = insn & 15;



		if (insn & 0x400)

		  *tp++ = '+';

	      }

	    *tp++ = ']';

	  }

	break;



      case 'x':

	tp = format_reg (disdata, (insn >> 12) & 15, tp, with_reg_prefix);

	*tp++ = '.';

	*tp++ = mode_char[(insn >> 4) & 3];

	break;



      case 'I':

	tp = format_dec (insn & 63, tp, 0);

	break;



      case 'b':

	{

	  int where = buffer[2] + buffer[3] * 256;



	  if (where > 32767)

	    where -= 65536;



	  where += addr + ((disdata->distype == cris_dis_v32) ? 0 : 4);



	  if (insn == BA_PC_INCR_OPCODE)

	    info->insn_type = dis_branch;

	  else

	    info->insn_type = dis_condbranch;



	  info->target = (bfd_vma) where;



	  *tp = 0;

	  tp = temp;

	  (*info->fprintf_func) (info->stream, "%s%s ",

				 temp, cris_cc_strings[insn >> 12]);



	  (*info->print_address_func) ((bfd_vma) where, info);

	}

      break;



    case 'c':

      tp = format_dec (insn & 31, tp, 0);

      break;



    case 'C':

      tp = format_dec (insn & 15, tp, 0);

      break;



    case 'o':

      {

	long offset = insn & 0xfe;

	bfd_vma target;



	if (insn & 1)

	  offset |= ~0xff;



	if (opcodep->match == BA_QUICK_OPCODE)

	  info->insn_type = dis_branch;

	else

	  info->insn_type = dis_condbranch;



	target = addr + ((disdata->distype == cris_dis_v32) ? 0 : 2) + offset;

	info->target = target;

	*tp = 0;

	tp = temp;

	(*info->fprintf_func) (info->stream, "%s", temp);

	(*info->print_address_func) (target, info);

      }

      break;



    case 'Q':

    case 'O':

      {

	long number = buffer[0];



	if (number > 127)

	  number = number - 256;



	tp = format_dec (number, tp, 1);

	*tp++ = ',';

	tp = format_reg (disdata, (insn >> 12) & 15, tp, with_reg_prefix);

      }

      break;



    case 'f':

      tp = print_flags (disdata, insn, tp);

      break;



    case 'i':

      tp = format_dec ((insn & 32) ? (insn & 31) | ~31L : insn & 31, tp, 1);

      break;



    case 'P':

      {

	const struct cris_spec_reg *sregp

	  = spec_reg_info ((insn >> 12) & 15, disdata->distype);



	if (sregp->name == NULL)

	  /* Should have been caught as a non-match earlier.  */

	  *tp++ = '?';

	else

	  {

	    if (with_reg_prefix)

	      *tp++ = REGISTER_PREFIX_CHAR;

	    strcpy (tp, sregp->name);

	    tp += strlen (tp);

	  }

      }

      break;



    default:

      strcpy (tp, "???");

      tp += 3;

    }

  }



  *tp = 0;



  if (prefix_opcodep)

    (*info->fprintf_func) (info->stream, " (OOPS unused prefix \"%s: %s\")",

			   prefix_opcodep->name, prefix_opcodep->args);



  (*info->fprintf_func) (info->stream, "%s", temp);



  /* Get info for matching case-tables, if we don't have any active.

     We assume that the last constant seen is used; either in the insn

     itself or in a "move.d const,rN, sub.d rN,rM"-like sequence.  */

  if (TRACE_CASE && case_offset_counter == 0)

    {

      if (CONST_STRNEQ (opcodep->name, "sub"))

	case_offset = last_immediate;



      /* It could also be an "add", if there are negative case-values.  */

      else if (CONST_STRNEQ (opcodep->name, "add"))

	/* The first case is the negated operand to the add.  */

	case_offset = -last_immediate;



      /* A bound insn will tell us the number of cases.  */

      else if (CONST_STRNEQ (opcodep->name, "bound"))

	no_of_case_offsets = last_immediate + 1;



      /* A jump or jsr or branch breaks the chain of insns for a

	 case-table, so assume default first-case again.  */

      else if (info->insn_type == dis_jsr

	       || info->insn_type == dis_branch

	       || info->insn_type == dis_condbranch)

	case_offset = 0;

    }

}
