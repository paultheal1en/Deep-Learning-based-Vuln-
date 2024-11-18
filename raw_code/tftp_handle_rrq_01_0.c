static void tftp_handle_rrq(struct tftp_t *tp, int pktlen)

{

  struct tftp_session *spt;

  int s, k, n;

  u_int8_t *src, *dst;



  s = tftp_session_allocate(tp);



  if (s < 0) {

    return;

  }



  spt = &tftp_sessions[s];



  src = tp->x.tp_buf;

  dst = spt->filename;

  n = pktlen - ((uint8_t *)&tp->x.tp_buf[0] - (uint8_t *)tp);



  /* get name */



  for (k = 0; k < n; k++) {

    if (k < TFTP_FILENAME_MAX) {

      dst[k] = src[k];

    }

    else {

      return;

    }



    if (src[k] == '\0') {

      break;

    }

  }



  if (k >= n) {

    return;

  }



  k++;



  /* check mode */

  if ((n - k) < 6) {

    return;

  }



  if (memcmp(&src[k], "octet\0", 6) != 0) {

      tftp_send_error(spt, 4, "Unsupported transfer mode", tp);

      return;

  }



  k += 6; /* skipping octet */



  /* do sanity checks on the filename */



  if ((spt->filename[0] != '/')

      || (spt->filename[strlen((char *)spt->filename) - 1] == '/')

      ||  strstr((char *)spt->filename, "/../")) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  /* only allow exported prefixes */



  if (!tftp_prefix) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  /* check if the file exists */



  if (tftp_read_data(spt, 0, spt->filename, 0) < 0) {

      tftp_send_error(spt, 1, "File not found", tp);

      return;

  }



  if (src[n - 1] != 0) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  while (k < n) {

      const char *key, *value;



      key = (char *)src + k;

      k += strlen(key) + 1;



      if (k >= n) {

	  tftp_send_error(spt, 2, "Access violation", tp);

	  return;

      }



      value = (char *)src + k;

      k += strlen(value) + 1;



      if (strcmp(key, "tsize") == 0) {

	  int tsize = atoi(value);

	  struct stat stat_p;



	  if (tsize == 0 && tftp_prefix) {

	      char buffer[1024];

	      int len;



	      len = snprintf(buffer, sizeof(buffer), "%s/%s",

			     tftp_prefix, spt->filename);



	      if (stat(buffer, &stat_p) == 0)

		  tsize = stat_p.st_size;

	      else {

		  tftp_send_error(spt, 1, "File not found", tp);

		  return;

	      }

	  }



	  tftp_send_oack(spt, "tsize", tsize, tp);

      }

  }



  tftp_send_data(spt, 1, tp);

}
