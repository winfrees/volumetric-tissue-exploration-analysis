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
package vtea.clustering;

import ij.IJ;
import java.util.ArrayList;
import javax.swing.JLabel;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import org.scijava.plugin.Plugin;
import smile.clustering.KMeans;
import vtea.featureprocessing.AbstractFeatureProcessing;
import vtea.featureprocessing.FeatureProcessing;

/**
 * K-Means Clustering by SMILE. 
 * 
 * @author winfrees
 */
@Plugin(type = FeatureProcessing.class)
public class KMeansClust extends AbstractFeatureProcessing {

    public static boolean validate = false;

    /**
     * Creates the Comment Text for the Block GUI.
     *
     * @param comComponents the parameters (Components) selected by the user in
     * the Setup Frame.
     * @return comment text detailing the parameters
     */
    public static String getBlockComment(ArrayList comComponents) {
        String comment = "<html>";
        comment = comment.concat(((JLabel) comComponents.get(4)).getText() + ": ");
        comment = comment.concat(((JSpinner) comComponents.get(5)).getValue().toString());
        comment = comment.concat(((JLabel) comComponents.get(6)).getText() + ": ");
        comment = comment.concat(((JTextField) comComponents.get(7)).getText());
        comment = comment.concat("</html>");
        return comment;
    }

    /**
     * Basic Constructor. Sets all protected variables
     */
    public KMeansClust() {
        VERSION = "0.1";
        AUTHOR = "Seth WInfree";
        COMMENT = "Implements the plugin from SMILE";
        NAME = "K-means Clustering";
        KEY = "Kmeans";
        TYPE = "Cluster";
    }

    /**
     * Constructor called for initialization of Setup GUI. When components are
     * added to this, the static method must be altered.
     *
     * @param max the number of objects segmented in the volume
     */
    public KMeansClust(int max) {
        this();

        protocol = new ArrayList();

        protocol.add(new JLabel("Clusters"));
        protocol.add(new JSpinner(new SpinnerNumberModel(5, 2, max, 1)));

        protocol.add(new JLabel("Iterations"));
        protocol.add(new JTextField("10", 2));
    }

    @Override
    public String getDataDescription(ArrayList params) {
        return KEY + '_' + String.valueOf((Integer) ((JSpinner) params.get(5)).getValue()) + '_' + getCurrentTime();
    }

    /**
     * Performs the X-Means clustering based on the parameters.
     *
     * @param al contains all of the parameters in the form of JComponents
     * @param feature the full data to be parsed and analyzed
     * @return true when complete
     */
    @Override
    public boolean process(ArrayList al, double[][] feature, boolean val) {
        int maxClust;
        int numTrials;
        int[] membership;

        ArrayList selectData = (ArrayList) al.get(1);
        boolean znorm = (boolean) al.get(0);
        feature = selectColumns(feature, selectData);
        feature = normalizeColumns(feature, znorm);
        dataResult.ensureCapacity(feature.length);

        JSpinner clust = (JSpinner) al.get(5);
        maxClust = ((Integer) clust.getValue());
        JTextField trial = (JTextField) al.get(7);
        numTrials = (Integer.parseInt(trial.getText()));
        
        

        IJ.log(String.format("PROFILING: Clustering using "  + this.NAME + " for a maximum of %d clusters", maxClust));
        long start = System.currentTimeMillis();
        KMeans xm = new KMeans(feature, maxClust, numTrials);
        membership = xm.getClusterLabel();
        IJ.log(String.format("PROFILING: Clustering completed in %d s, %d clusters found", (System.currentTimeMillis() - start) / 1000, xm.getNumClusters()));

        //Exists so that MicroExplorer deals with same data structure for both Clustering and Reduction
        ArrayList holder = new ArrayList();
        for (int m : membership) {
            holder.add(m);
        }
        dataResult.add(holder);

        return true;
    }

}
