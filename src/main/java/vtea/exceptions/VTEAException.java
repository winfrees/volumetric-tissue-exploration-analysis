/*
 * Copyright (C) 2020 Indiana University
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 */
package vtea.exceptions;

/**
 * Base exception class for all VTEA-specific exceptions.
 *
 * This exception hierarchy allows for better error handling and clearer
 * error messages throughout the VTEA application.
 *
 * @author VTEA Development Team
 */
public class VTEAException extends Exception {

    private static final long serialVersionUID = 1L;

    /**
     * Constructs a new VTEA exception with null as its detail message.
     */
    public VTEAException() {
        super();
    }

    /**
     * Constructs a new VTEA exception with the specified detail message.
     *
     * @param message the detail message
     */
    public VTEAException(String message) {
        super(message);
    }

    /**
     * Constructs a new VTEA exception with the specified detail message and cause.
     *
     * @param message the detail message
     * @param cause the cause of this exception
     */
    public VTEAException(String message, Throwable cause) {
        super(message, cause);
    }

    /**
     * Constructs a new VTEA exception with the specified cause.
     *
     * @param cause the cause of this exception
     */
    public VTEAException(Throwable cause) {
        super(cause);
    }
}
