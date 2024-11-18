print_insn_ppi (int field_b, struct disassemble_info *info)

{

  static const char *sx_tab[] = { "x0", "x1", "a0", "a1" };

  static const char *sy_tab[] = { "y0", "y1", "m0", "m1" };

  fprintf_ftype fprintf_fn = info->fprintf_func;

  void *stream = info->stream;

  unsigned int nib1, nib2, nib3;

  unsigned int altnib1, nib4;

  const char *dc = NULL;

  const sh_opcode_info *op;



  if ((field_b & 0xe800) == 0)

    {

      fprintf_fn (stream, "psh%c\t#%d,",

		  field_b & 0x1000 ? 'a' : 'l',

		  (field_b >> 4) & 127);

      print_dsp_reg (field_b & 0xf, fprintf_fn, stream);

      return;

    }

  if ((field_b & 0xc000) == 0x4000 && (field_b & 0x3000) != 0x1000)

    {

      static const char *du_tab[] = { "x0", "y0", "a0", "a1" };

      static const char *se_tab[] = { "x0", "x1", "y0", "a1" };

      static const char *sf_tab[] = { "y0", "y1", "x0", "a1" };

      static const char *sg_tab[] = { "m0", "m1", "a0", "a1" };



      if (field_b & 0x2000)

	{

	  fprintf_fn (stream, "p%s %s,%s,%s\t",

		      (field_b & 0x1000) ? "add" : "sub",

		      sx_tab[(field_b >> 6) & 3],

		      sy_tab[(field_b >> 4) & 3],

		      du_tab[(field_b >> 0) & 3]);

	}

      else if ((field_b & 0xf0) == 0x10

	       && info->mach != bfd_mach_sh_dsp

	       && info->mach != bfd_mach_sh3_dsp)

	{

	  fprintf_fn (stream, "pclr %s \t", du_tab[(field_b >> 0) & 3]);

	}

      else if ((field_b & 0xf3) != 0)

	{

	  fprintf_fn (stream, ".word 0x%x\t", field_b);

	}

      fprintf_fn (stream, "pmuls%c%s,%s,%s",

		  field_b & 0x2000 ? ' ' : '\t',

		  se_tab[(field_b >> 10) & 3],

		  sf_tab[(field_b >>  8) & 3],

		  sg_tab[(field_b >>  2) & 3]);

      return;

    }



  nib1 = PPIC;

  nib2 = field_b >> 12 & 0xf;

  nib3 = field_b >> 8 & 0xf;

  nib4 = field_b >> 4 & 0xf;

  switch (nib3 & 0x3)

    {

    case 0:

      dc = "";

      nib1 = PPI3;

      break;

    case 1:

      dc = "";

      break;

    case 2:

      dc = "dct ";

      nib3 -= 1;

      break;

    case 3:

      dc = "dcf ";

      nib3 -= 2;

      break;

    }

  if (nib1 == PPI3)

    altnib1 = PPI3NC;

  else

    altnib1 = nib1;

  for (op = sh_table; op->name; op++)

    {

      if ((op->nibbles[1] == nib1 || op->nibbles[1] == altnib1)

	  && op->nibbles[2] == nib2

	  && op->nibbles[3] == nib3)

	{

	  int n;



	  switch (op->nibbles[4])

	    {

	    case HEX_0:

	      break;

	    case HEX_XX00:

	      if ((nib4 & 3) != 0)

		continue;

	      break;

	    case HEX_1:

	      if ((nib4 & 3) != 1)

		continue;

	      break;

	    case HEX_00YY:

	      if ((nib4 & 0xc) != 0)

		continue;

	      break;

	    case HEX_4:

	      if ((nib4 & 0xc) != 4)

		continue;

	      break;

	    default:

	      abort ();

	    }

	  fprintf_fn (stream, "%s%s\t", dc, op->name);

	  for (n = 0; n < 3 && op->arg[n] != A_END; n++)

	    {

	      if (n && op->arg[1] != A_END)

		fprintf_fn (stream, ",");

	      switch (op->arg[n])

		{

		case DSP_REG_N:

		  print_dsp_reg (field_b & 0xf, fprintf_fn, stream);

		  break;

		case DSP_REG_X:

		  fprintf_fn (stream, sx_tab[(field_b >> 6) & 3]);

		  break;

		case DSP_REG_Y:

		  fprintf_fn (stream, sy_tab[(field_b >> 4) & 3]);

		  break;

		case A_MACH:

		  fprintf_fn (stream, "mach");

		  break;

		case A_MACL:

		  fprintf_fn (stream, "macl");

		  break;

		default:

		  abort ();

		}

	    }

	  return;

	}

    }

  /* Not found.  */

  fprintf_fn (stream, ".word 0x%x", field_b);

}
