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
package vteaexploration;

import ij.IJ;
import java.awt.Color;
import java.awt.Component;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;
import java.util.ListIterator;
import javax.swing.AbstractCellEditor;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JColorChooser;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JTable;
import javax.swing.border.Border;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.DefaultTableModel;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import javax.swing.table.TableColumn;
import javax.swing.table.TableModel;
import vtea.exploration.listeners.NameUpdateListener;
import vtea.exploration.listeners.colorUpdateListener;
import vtea.exploration.listeners.remapOverlayListener;
import vtea.exploration.listeners.GateManagerActionListener;
import vtea.exploration.plotgatetools.gates.PolygonGate;

/**
 *
 * @author vinfrais
 */
public class TableWindow extends javax.swing.JFrame implements TableModelListener {

    private Object[][] DataTableArray = new Object[4][15];

    private ArrayList<NameUpdateListener> nameUpdateListeners = new ArrayList<>();
    private ArrayList<remapOverlayListener> remapOverlayListeners = new ArrayList<>();
    private ArrayList<colorUpdateListener> UpdateColorListeners = new ArrayList<>();
    
    private ArrayList<GateManagerActionListener> gateManagerListeners = new ArrayList<>();
    
    private ArrayList<PolygonGate> gateList = new ArrayList();

    /**
     * Creates new form gatePercentages
     */
    public TableWindow(String name) {
        initComponents();
        GateDataTable.getModel().addTableModelListener(this);
        setTitle(getTitle()+": "+name);
    }
    
    public void setVisible() {
        this.setVisible(false);
    }

    private void notifyUpdateNameListeners(String name, int row) {
        for (NameUpdateListener listener : nameUpdateListeners) {
            listener.onUpdateName(name, row);
        }
    }

    public void addUpdateNameListener(NameUpdateListener listener) {
        nameUpdateListeners.add(listener);
    }

    private void notifyRemapOverlayListeners(boolean b, int row) {
        for (remapOverlayListener listener : remapOverlayListeners) {
            listener.onRemapOverlay(b, row);
        }
    }

    public void addRemapOverlayListener(remapOverlayListener listener) {
        remapOverlayListeners.add(listener);
    }

    private void notifyUpdateColorListeners(Color color, int row) {
        for (colorUpdateListener listener : UpdateColorListeners) {
            listener.onColorUpdate(color, row);
        }
    }
    
    public void addGateActionListener(GateManagerActionListener listener) {
        gateManagerListeners.add(listener);
    }

    private void notifyGateActionListeners(String st) {
        for (GateManagerActionListener listener : gateManagerListeners) {
            listener.doGates(st);
        }
    }

    public void addUpdateColorListener(colorUpdateListener listener) {
        UpdateColorListeners.add(listener);
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {
        java.awt.GridBagConstraints gridBagConstraints;

        jRadioButton1 = new javax.swing.JRadioButton();
        jPanel1 = new javax.swing.JPanel();
        currentMeasure = new javax.swing.JLabel();
        jToolBar1 = new javax.swing.JToolBar();
        addMeasurement = new javax.swing.JButton();
        jSeparator2 = new javax.swing.JToolBar.Separator();
        exportGates = new javax.swing.JButton();
        LoadGates = new javax.swing.JButton();
        jSeparator1 = new javax.swing.JToolBar.Separator();
        jButton1 = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        GateDataTable = new javax.swing.JTable();

        jRadioButton1.setText("jRadioButton1");

        setTitle("Gate Management");
        setAlwaysOnTop(true);
        setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
        setFocusTraversalPolicyProvider(true);
        setMaximumSize(new java.awt.Dimension(725, 270));
        setMinimumSize(new java.awt.Dimension(725, 270));
        setPreferredSize(new java.awt.Dimension(725, 270));
        setResizable(false);
        setSize(new java.awt.Dimension(725, 280));
        getContentPane().setLayout(new java.awt.GridBagLayout());

        jPanel1.setMaximumSize(new java.awt.Dimension(700, 40));
        jPanel1.setMinimumSize(new java.awt.Dimension(700, 40));
        jPanel1.setPreferredSize(new java.awt.Dimension(700, 40));
        jPanel1.setRequestFocusEnabled(false);
        jPanel1.setLayout(new java.awt.GridBagLayout());

        currentMeasure.setFont(new java.awt.Font("Lucida Grande", 0, 14)); // NOI18N
        currentMeasure.setText("currentMeasure");
        currentMeasure.setToolTipText(currentMeasure.getText());
        currentMeasure.setHorizontalTextPosition(javax.swing.SwingConstants.RIGHT);
        currentMeasure.setMaximumSize(new java.awt.Dimension(670, 20));
        currentMeasure.setMinimumSize(new java.awt.Dimension(520, 20));
        currentMeasure.setPreferredSize(new java.awt.Dimension(580, 20));
        jPanel1.add(currentMeasure, new java.awt.GridBagConstraints());

        jToolBar1.setFloatable(false);
        jToolBar1.setRollover(true);
        jToolBar1.setBorderPainted(false);
        jToolBar1.setPreferredSize(new java.awt.Dimension(156, 35));

        addMeasurement.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/list-add-3 2.png"))); // NOI18N
        addMeasurement.setToolTipText("Add measurement to ImageJ log.");
        addMeasurement.setFocusable(false);
        addMeasurement.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        addMeasurement.setMaximumSize(new java.awt.Dimension(35, 40));
        addMeasurement.setMinimumSize(new java.awt.Dimension(35, 40));
        addMeasurement.setPreferredSize(new java.awt.Dimension(35, 40));
        addMeasurement.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        addMeasurement.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                addMeasurementActionPerformed(evt);
            }
        });
        jToolBar1.add(addMeasurement);
        jToolBar1.add(jSeparator2);

        exportGates.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/document-save-2_24.png"))); // NOI18N
        exportGates.setToolTipText("Save gates...");
        exportGates.setFocusable(false);
        exportGates.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        exportGates.setMaximumSize(new java.awt.Dimension(35, 40));
        exportGates.setMinimumSize(new java.awt.Dimension(35, 40));
        exportGates.setName(""); // NOI18N
        exportGates.setPreferredSize(new java.awt.Dimension(35, 40));
        exportGates.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        exportGates.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                exportGatesActionPerformed(evt);
            }
        });
        jToolBar1.add(exportGates);

        LoadGates.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/document-open-folder_24.png"))); // NOI18N
        LoadGates.setToolTipText("Load gates...");
        LoadGates.setFocusable(false);
        LoadGates.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        LoadGates.setMaximumSize(new java.awt.Dimension(35, 40));
        LoadGates.setMinimumSize(new java.awt.Dimension(35, 40));
        LoadGates.setName(""); // NOI18N
        LoadGates.setPreferredSize(new java.awt.Dimension(35, 40));
        LoadGates.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        LoadGates.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                LoadGatesActionPerformed(evt);
            }
        });
        jToolBar1.add(LoadGates);
        jToolBar1.add(jSeparator1);

        jButton1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/icons/Heatmap.png"))); // NOI18N
        jButton1.setToolTipText("Make a heatmap for a given feature.");
        jButton1.setFocusable(false);
        jButton1.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        jButton1.setMargin(new java.awt.Insets(0, 1, 0, 1));
        jButton1.setMaximumSize(new java.awt.Dimension(35, 40));
        jButton1.setMinimumSize(new java.awt.Dimension(35, 40));
        jButton1.setPreferredSize(new java.awt.Dimension(35, 40));
        jButton1.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        jToolBar1.add(jButton1);

        jPanel1.add(jToolBar1, new java.awt.GridBagConstraints());

        getContentPane().add(jPanel1, new java.awt.GridBagConstraints());

        jScrollPane1.setMaximumSize(new java.awt.Dimension(700, 200));
        jScrollPane1.setMinimumSize(new java.awt.Dimension(700, 200));
        jScrollPane1.setPreferredSize(new java.awt.Dimension(700, 200));
        jScrollPane1.setSize(new java.awt.Dimension(700, 200));

        GateDataTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null},
                {null, null, null, null, null, null, null, null, null}
            },
            new String [] {
                "Gate", "Name", "X axis", "Y Axis", "Gated", "Total", "%", "", ""
            }
        ) {
            Class[] types = new Class [] {
                java.lang.String.class, java.lang.String.class, java.lang.String.class, java.lang.Object.class, java.lang.String.class, java.lang.String.class, java.lang.String.class, java.lang.Object.class, java.lang.Object.class
            };
            boolean[] canEdit = new boolean [] {
                false, true, false, false, false, false, false, false, false
            };

            public Class getColumnClass(int columnIndex) {
                return types [columnIndex];
            }

            public boolean isCellEditable(int rowIndex, int columnIndex) {
                return canEdit [columnIndex];
            }
        });
        GateDataTable.setAutoResizeMode(javax.swing.JTable.AUTO_RESIZE_OFF);
        GateDataTable.setMaximumSize(new java.awt.Dimension(630, 200));
        GateDataTable.setMinimumSize(new java.awt.Dimension(630, 200));
        GateDataTable.setPreferredSize(new java.awt.Dimension(640, 200));
        jScrollPane1.setViewportView(GateDataTable);

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 1;
        getContentPane().add(jScrollPane1, gridBagConstraints);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void addMeasurementActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_addMeasurementActionPerformed
       IJ.log(this.getTitle());
       System.out.println(this.getTitle());
       
        ListIterator<PolygonGate> itr = gateList.listIterator();

            while (itr.hasNext()) {
                PolygonGate pg = (PolygonGate) itr.next();
                if(pg.getSelected()){
//                 System.out.println("Gating results: Name: " + pg.getName() + ", " +
//                  pg.getColor().toString() + ", " + pg.getXAxis()  + ", " + pg.getYAxis() + ", " +
//                  pg.getObjectsInGate() + ", " + pg.getTotalObjects() + ", " +
//                  (float) 100 * ((int) pg.getObjectsInGate()) / ((int) pg.getTotalObjects())); 
//                 
                 IJ.log("Gating results: Name: " + pg.getName() + ", " +
                  pg.getColor().toString() + ", " + pg.getXAxis()  + ", " + pg.getYAxis() + ", " +
                  pg.getObjectsInGate() + ", " + pg.getTotalObjects() + ", " +
                  (float) 100 * ((int) pg.getObjectsInGate()) / ((int) pg.getTotalObjects()));
                }
            }
    }//GEN-LAST:event_addMeasurementActionPerformed

    private void exportGatesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_exportGatesActionPerformed
    notifyGateActionListeners("export");
    }//GEN-LAST:event_exportGatesActionPerformed

    private void LoadGatesActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_LoadGatesActionPerformed
    notifyGateActionListeners("import");
    }//GEN-LAST:event_LoadGatesActionPerformed

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JTable GateDataTable;
    private javax.swing.JButton LoadGates;
    private javax.swing.JButton addMeasurement;
    public javax.swing.JLabel currentMeasure;
    private javax.swing.JButton exportGates;
    private javax.swing.JButton jButton1;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JRadioButton jRadioButton1;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JToolBar.Separator jSeparator1;
    private javax.swing.JToolBar.Separator jSeparator2;
    private javax.swing.JToolBar jToolBar1;
    // End of variables declaration//GEN-END:variables

    public void setMeasurementsText(String st) {
        if (st.length() > 60) {
            this.currentMeasure.setFont(new java.awt.Font("Lucida Grande", 0, 14));
        } else if (st.length() < 60 && st.length() > 30) {
            this.currentMeasure.setFont(new java.awt.Font("Lucida Grande", 0, 16));
        } else {
            this.currentMeasure.setFont(new java.awt.Font("Lucida Grande", 0, 18));
        }
        this.currentMeasure.setToolTipText(st);
        this.currentMeasure.setText(st);
    }
    
    public void addGateToTable(PolygonGate g) {
        //if (gateList.size() > 0) {
            if (g.getGateAsPoints().size() > 2) {
                ((DefaultTableModel) GateDataTable.getModel()).addRow(
                        new Object[]{g.getSelected(),
                            g.getColor(),
                            g.getName(),
                            g.getXAxis(),
                            g.getYAxis(),
                            g.getObjectsInGate(),
                            g.getTotalObjects(),
                            (float) 100 * ((int) g.getObjectsInGate()) / ((int) g.getTotalObjects())
                        });
//            } else {
//                ArrayList<PolygonGate> gates = new ArrayList<>();
//                gates.add(g);
//                updateTable(gates);
                
          //  }
//        GateDataTable.getModel().setValueAt(g.getSelected(), gateList.size(), 0);
//        GateDataTable.getModel().setValueAt(g.getColor(), gateList.size(), 1);
//        GateDataTable.getModel().setValueAt(g.getName(), gateList.size(), 2);
//        GateDataTable.getModel().setValueAt(g.getXAxis(), gateList.size(), 3);
//        GateDataTable.getModel().setValueAt(g.getYAxis(),  gateList.size(), 4);
//        GateDataTable.getModel().setValueAt(g.getObjectsInGate(),  gateList.size(), 5);
//        GateDataTable.getModel().setValueAt(g.getTotalObjects(),  gateList.size(), 6);
//        GateDataTable.getModel().setValueAt((float) 100 * ((int) g.getObjectsInGate()) / ((int) g.getTotalObjects()),gateList.size(), 7);
            
        }
            pack();
            repaint();
            
    }
    
    public void updateGateSelection(ArrayList<PolygonGate> gates){
        
        ListIterator<PolygonGate> itr = gates.listIterator();
        int i = 0;
         while (itr.hasNext()) {
             PolygonGate pg = (PolygonGate) itr.next();
             boolean selected = pg.getSelected();
             GateDataTable.getModel().setValueAt(selected, i, 0);
             i++;
         }
            pack();
        repaint();
        
    }
    
    private ArrayList<PolygonGate> cleanGateList(ArrayList<PolygonGate> gates){
        ListIterator<PolygonGate> itr = gates.listIterator();
        
        ArrayList<PolygonGate> result = new ArrayList<>();

            int i = 0;
            while (itr.hasNext()) {
                PolygonGate g = itr.next();
                if(g.getGateAsPoints().size() > 2){
                    result.add(g);
                }
            }
            return result;
    }
    
       

    public void updateTable(ArrayList<PolygonGate> gates) {
        
        gateList = cleanGateList(gates);
        

        // System.out.println("PROFILING:  Rebuilding GM with gates: " + gates.size());
        if (gateList.size() > 0) {

            ListIterator<PolygonGate> itr = gateList.listIterator();

            Object[][] gatesData = new Object[gateList.size()][9];
            int i = 0;
            while (itr.hasNext()) {
                Object[] gateData = new Object[9];

                PolygonGate pg = (PolygonGate) itr.next();
                
                

                gateData[0] = pg.getSelected();
                gateData[1] = pg.getColor();
                gateData[2] = pg.getName();
                gateData[3] = pg.getXAxis();
                gateData[4] = pg.getYAxis();
                gateData[5] = pg.getObjectsInGate();
                gateData[6] = pg.getTotalObjects();
                if (pg.getTotalObjects() > 0) {
                    gateData[7] = (float) 100 * ((int) pg.getObjectsInGate()) / ((int) pg.getTotalObjects());
                }

                gatesData[i] = gateData;

                i++;
   
                
            }

            String[] columnNames = {"View",
                "Color",
                "Name",
                "XAxis",
                "YAxis",
                "Gated",
                "Total",
                "%"
            };

            this.GateDataTable = new JTable(gatesData, columnNames) {
                @Override
                public TableCellRenderer getCellRenderer(int row, int column) {
                    for (int i = 0; i < gatesData.length; i++) {
                        if ((row == i) && (column == 1)) {
                            return new ColorRenderer(true, (Color) gatesData[i][1]);
                        }
                    }

                    return super.getCellRenderer(row, column);
                }
            };

            GateDataTable.setModel(new javax.swing.table.DefaultTableModel(
                    gatesData,
                    columnNames
            ) {

                public Class getColumnClass(int c) {
                    return getValueAt(0, c).getClass();
                }

                public boolean isCellEditable(int row, int col) {

                    if (col == 2 || col == 0) {
                        return true;
                    } else {
                        return false;
                    }
                }

            });

            GateDataTable.setDefaultRenderer(Color.class,
                    new ColorRenderer(true, new Color(255, 0, 0)));
            GateDataTable.setDefaultEditor(Color.class,
                    new ColorEditor());

            GateDataTable.getModel().addTableModelListener(this);

            GateDataTable.setAutoResizeMode(javax.swing.JTable.AUTO_RESIZE_OFF);
            GateDataTable.setMaximumSize(new java.awt.Dimension(680, 1400));
            GateDataTable.setMinimumSize(new java.awt.Dimension(630, 1400));
            GateDataTable.setPreferredSize(new java.awt.Dimension(680, 1400));
            GateDataTable.setShowGrid(true);

            TableColumn column = null;
            column = GateDataTable.getColumnModel().getColumn(0);
            column.setPreferredWidth(40);
            column = GateDataTable.getColumnModel().getColumn(1);
            column.setPreferredWidth(60);
            column = GateDataTable.getColumnModel().getColumn(2);
            column.setPreferredWidth(120);
            column = GateDataTable.getColumnModel().getColumn(3);
            column.setPreferredWidth(115);
            column = GateDataTable.getColumnModel().getColumn(4);
            column.setPreferredWidth(115);
            column = GateDataTable.getColumnModel().getColumn(5);
            column.setPreferredWidth(90);
            column = GateDataTable.getColumnModel().getColumn(6);
            column.setPreferredWidth(90);
            column = GateDataTable.getColumnModel().getColumn(7);
            column.setPreferredWidth(55);
            GateDataTable.doLayout();
            GateDataTable.repaint();

            jScrollPane1.setViewportView(GateDataTable);
        } else {
            GateDataTable = new JTable();
            GateDataTable.setModel(new javax.swing.table.DefaultTableModel(
                    new Object[][]{
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null},
                        {null, null, null, null, null, null, null, null}
                    },
                    new String[]{
                        "View",
                        "Color",
                        "Name",
                        "XAxis",
                        "YAxis",
                        "Gated",
                        "Total",
                        "%"

                    }
            ) {
                Class[] types = new Class[]{
                    java.lang.String.class, java.lang.String.class, java.lang.String.class, java.lang.Object.class, java.lang.String.class, java.lang.String.class, java.lang.String.class, java.lang.Object.class, java.lang.Object.class
                };
                boolean[] canEdit = new boolean[]{
                    false, true, false, false, false, false, false, false, false
                };

                public Class getColumnClass(int columnIndex) {
                    return types[columnIndex];
                }

                public boolean isCellEditable(int rowIndex, int columnIndex) {
                    return canEdit[columnIndex];
                }
            });

            jScrollPane1.setViewportView(GateDataTable);
        }
        

    }

    @Override
    public void tableChanged(TableModelEvent e) {
        int row = e.getFirstRow();
        int column = e.getColumn();
        TableModel model = (TableModel) e.getSource();
        String columnName = model.getColumnName(column);
        Object data = model.getValueAt(row, column);
        if (column == 2) {
            notifyUpdateNameListeners((String) data, row);
        }
        if (column == 0) {
            notifyRemapOverlayListeners((Boolean) data, row);
        }
        if (column == 1) {
            notifyUpdateColorListeners((Color) data, row);
        }
        
        // System.out.println("DEBUGGING: Gate Percentages, tableChanged" + e.toString());
    }

}

class ColorRenderer extends JLabel
        implements TableCellRenderer {

    Border unselectedBorder = null;
    Border selectedBorder = null;
    boolean isBordered = true;
    Color color = new Color(255, 0, 0, 0);

    public ColorRenderer(boolean isBordered, Color color) {
        this.isBordered = isBordered;
        this.color = color;
        setOpaque(true); //MUST do this for background to show up.
    }

    public Component getTableCellRendererComponent(
            JTable table, Object color,
            boolean isSelected, boolean hasFocus,
            int row, int column) {
        Color newColor = (Color) color;
        setBackground(newColor);
        if (isBordered) {
            if (isSelected) {
                if (selectedBorder == null) {
                    selectedBorder = BorderFactory.createMatteBorder(2, 5, 2, 5,
                            table.getSelectionBackground());
                }
                setBorder(selectedBorder);
            } else {
                if (unselectedBorder == null) {
                    unselectedBorder = BorderFactory.createMatteBorder(2, 5, 2, 5,
                            table.getBackground());
                }
                setBorder(unselectedBorder);
            }
        }

        setToolTipText("RGB value: " + newColor.getRed() + ", "
                + newColor.getGreen() + ", "
                + newColor.getBlue());
        return this;
    }
}

class ColorEditor extends AbstractCellEditor
        implements TableCellEditor,
        ActionListener {

    protected static final String EDIT = "edit";
    Color currentColor;
    JButton button;
    JColorChooser colorChooser;
    JDialog dialog;

    public ColorEditor() {
        //Set up the editor (from the table's point of view),
        //which is a button.
        //This button brings up the color chooser dialog,
        //which is the editor from the user's point of view.
        button = new JButton();
        button.setActionCommand(EDIT);
        button.addActionListener(this);
        button.setBorderPainted(false);

        //Set up the dialog that the button brings up.
        colorChooser = new JColorChooser();
        dialog = JColorChooser.createDialog(button,
                "Pick a Color",
                true, //modal
                colorChooser,
                this, //OK button handler
                null); //no CANCEL button handler
    }

    /**
     * Handles events from the editor button and from the dialog's OK button.
     */
    public void actionPerformed(ActionEvent e) {
        if (EDIT.equals(e.getActionCommand())) {
            //The user has clicked the cell, so
            //bring up the dialog.
            button.setBackground(currentColor);
            colorChooser.setColor(currentColor);
            dialog.setVisible(true);

            //Make the renderer reappear.
            fireEditingStopped();

        } else { //User pressed dialog's "OK" button.
            currentColor = colorChooser.getColor();
        }
    }

    //Implement the one CellEditor method that AbstractCellEditor doesn't.
    public Object getCellEditorValue() {
        return currentColor;
    }

    //Implement the one method defined by TableCellEditor.
    public Component getTableCellEditorComponent(JTable table,
            Object value,
            boolean isSelected,
            int row,
            int column) {
        currentColor = (Color) value;
        return button;
    }
}
