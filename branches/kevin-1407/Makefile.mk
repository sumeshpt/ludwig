##############################################################################
#
#  Makefile.mk
#
#  Define here platform-dependent information.
#  There are no targets.
#
##############################################################################

ROOT_DIR := $(dir $(lastword $(MAKEFILE_LIST)))

include $(ROOT_DIR)/config.mk
