/*
 * Copyright (C) 2021 SciJava
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
package vtea.plotprocessing;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.script.ScriptEngine;
import javax.script.ScriptException;
import org.renjin.script.RenjinScriptEngineFactory;
import org.scijava.plugin.Plugin;
import static vtea.renjin.AbstractRenjin.VTEACOLORS;

/**
 *
 * @author sethwinfree
 */
@Plugin(type = PlotMaker.class)
public class ViolinPlot extends AbstractPlotMaker {

    ViolinPlot() {

        VERSION = "0.1";
        AUTHOR = "Seth Winfree";
        COMMENT = "Generates violin plots.";
        NAME = "Violin Plot";
        KEY = "ViolinPlot";

    }

    @Override
    public String makePlot(ArrayList<ArrayList> al, String location, String key, String feature, String group) {
       
        
        String filename = key + "_" + System.currentTimeMillis();

        ArrayList<Number> objects = (ArrayList) al.get(0);
        ArrayList<Number> features = (ArrayList) al.get(1);
        ArrayList<Number> groups = (ArrayList) al.get(2);

        try {
            File file = new File(location + System.getProperty("file.separator") + filename);
            PrintWriter pw;

            pw = new PrintWriter(file);

            StringBuilder sb = new StringBuilder();

            sb.append("Object");
            sb.append(',');
            sb.append(feature);
            sb.append(',');
            sb.append(group);
            sb.append('\n');
            for (int i = 0; i < objects.size(); i++) {
                sb.append(features.get(i));
                sb.append(",");
                sb.append(groups.get(i));
                sb.append('\n');
            }

            pw.write(sb.toString());
            pw.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(ViolinPlot.class.getName()).log(Level.SEVERE, null, ex);
        }
        try {
            RenjinScriptEngineFactory factory = new RenjinScriptEngineFactory();

            ScriptEngine engine = factory.getScriptEngine();
            
            engine.eval("library(ggplot2)");
            engine.eval("library(colorspace)");
            engine.eval("plot <- read.csv('" + location + 
                    System.getProperty("file.separator") + filename + ".csv')");
            engine.eval(VTEACOLORS);
            engine.eval("out <- ggplot(plot, aes(factor(plot$Gmeans_5__15_33_0),"
               + " plot$Ch_2_Mean, fill = as.factor(plot$Gmeans_5__15_33_0)))");
            engine.eval("out <- out + geom_violin(scale = \"width\", alpha = 0.5) "
                    + "+ theme_bw() + theme(legend.position = \"none\") + "
                    + "scale_fill_manual(values = plot_colors)");
            
            engine.eval("png('" + vtea._vtea.PLOT_DIRECTORY + System.getProperty("file.separator") + filename + ".png')");
            engine.eval("print(out)");
            engine.eval("dev.off()");

        } catch (ScriptException ex) {
            Logger.getLogger(ViolinPlot.class.getName()).log(Level.SEVERE, null, ex);
        }
        return filename + ".png";
    }

}
