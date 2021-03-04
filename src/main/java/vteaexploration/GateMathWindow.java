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
package vteaexploration;

import ij.gui.Roi;
import ij.plugin.frame.RoiManager;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.util.ArrayList;
import java.util.ListIterator;
import javax.swing.DefaultCellEditor;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTable;
import javax.swing.JTree;
import javax.swing.SpinnerNumberModel;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.event.TreeSelectionEvent;
import javax.swing.event.TreeSelectionListener;
import javax.swing.table.TableColumn;
import javax.swing.table.TableModel;
import javax.swing.tree.DefaultMutableTreeNode;
import javax.swing.tree.TreeModel;
import static vtea._vtea.GATEMATHMAP;
import static vtea._vtea.GATEMATHOPTIONS;
import vtea.exploration.listeners.AddClassByMathListener;
import vtea.exploration.plotgatetools.gates.PolygonGate;

/**
 *
 * @author sethwinfree
 */
public class GateMathWindow extends javax.swing.JFrame implements TableModelListener {

    ArrayList<MicroExplorer> subgated = new ArrayList<>();
    String parentName = "untitled";
    String[] operations = {"AND", "OR", "XOR", ""};

    ArrayList<String> gateNames = new ArrayList<>();
    ArrayList<String> opNames = new ArrayList<>();

    ArrayList<AddClassByMathListener> MathListeners = new ArrayList<AddClassByMathListener>();

    /**
     * Creates new form GatingHierarchy
     */
    public GateMathWindow(ArrayList<PolygonGate> gates) {
        initComponents();
        updateTable(gates);
    }

    public void addMicroExplorer(MicroExplorer me) {
        subgated.add(me);
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

        jScrollPane = new javax.swing.JScrollPane();
        GateMathTable = new javax.swing.JTable();
        classPanel = new javax.swing.JPanel();
        jPanel4 = new javax.swing.JPanel();
        jLabel1 = new javax.swing.JLabel();
        classSpinner = new javax.swing.JSpinner();
        ButtonPanel = new javax.swing.JPanel();
        jPanel3 = new javax.swing.JPanel();
        Cancel = new javax.swing.JButton();
        Calculate = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.DISPOSE_ON_CLOSE);
        setTitle("Gating math");
        setAlwaysOnTop(true);
        setMaximumSize(new java.awt.Dimension(400, 300));
        setMinimumSize(new java.awt.Dimension(400, 300));
        setPreferredSize(new java.awt.Dimension(400, 300));

        GateMathTable.setModel(new javax.swing.table.DefaultTableModel(
            new Object [][] {
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null},
                {null, null, null, null}
            },
            new String [] {
                "Title 1", "Title 2", "Title 3", "Title 4"
            }
        ));
        jScrollPane.setViewportView(GateMathTable);

        getContentPane().add(jScrollPane, java.awt.BorderLayout.CENTER);

        classPanel.setMaximumSize(new java.awt.Dimension(400, 32));
        classPanel.setMinimumSize(new java.awt.Dimension(400, 32));
        classPanel.setPreferredSize(new java.awt.Dimension(400, 32));
        classPanel.setLayout(new java.awt.GridBagLayout());

        jPanel4.setMaximumSize(new java.awt.Dimension(200, 50));
        jPanel4.setMinimumSize(new java.awt.Dimension(200, 50));

        javax.swing.GroupLayout jPanel4Layout = new javax.swing.GroupLayout(jPanel4);
        jPanel4.setLayout(jPanel4Layout);
        jPanel4Layout.setHorizontalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 200, Short.MAX_VALUE)
        );
        jPanel4Layout.setVerticalGroup(
            jPanel4Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 2;
        gridBagConstraints.gridy = 0;
        classPanel.add(jPanel4, gridBagConstraints);

        jLabel1.setText("Add to class");
        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 0;
        classPanel.add(jLabel1, gridBagConstraints);

        classSpinner.setModel(new SpinnerNumberModel(0, -1, 100, 1));
        classSpinner.setMaximumSize(new java.awt.Dimension(80, 30));
        classSpinner.setMinimumSize(new java.awt.Dimension(80, 30));
        classSpinner.setPreferredSize(new java.awt.Dimension(80, 30));
        classSpinner.setSize(new java.awt.Dimension(80, 30));
        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 1;
        gridBagConstraints.gridy = 0;
        classPanel.add(classSpinner, gridBagConstraints);

        getContentPane().add(classPanel, java.awt.BorderLayout.PAGE_START);

        ButtonPanel.setMaximumSize(new java.awt.Dimension(400, 32));
        ButtonPanel.setMinimumSize(new java.awt.Dimension(400, 32));
        ButtonPanel.setPreferredSize(new java.awt.Dimension(400, 32));
        ButtonPanel.setLayout(new java.awt.GridBagLayout());

        jPanel3.setMaximumSize(new java.awt.Dimension(200, 50));
        jPanel3.setMinimumSize(new java.awt.Dimension(200, 50));
        jPanel3.setPreferredSize(new java.awt.Dimension(200, 50));

        javax.swing.GroupLayout jPanel3Layout = new javax.swing.GroupLayout(jPanel3);
        jPanel3.setLayout(jPanel3Layout);
        jPanel3Layout.setHorizontalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 200, Short.MAX_VALUE)
        );
        jPanel3Layout.setVerticalGroup(
            jPanel3Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 50, Short.MAX_VALUE)
        );

        gridBagConstraints = new java.awt.GridBagConstraints();
        gridBagConstraints.gridx = 0;
        gridBagConstraints.gridy = 0;
        ButtonPanel.add(jPanel3, gridBagConstraints);

        Cancel.setText("Cancel");
        Cancel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                CancelActionPerformed(evt);
            }
        });
        ButtonPanel.add(Cancel, new java.awt.GridBagConstraints());

        Calculate.setText("Calculate");
        Calculate.setEnabled(false);
        Calculate.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                CalculateActionPerformed(evt);
            }
        });
        ButtonPanel.add(Calculate, new java.awt.GridBagConstraints());

        getContentPane().add(ButtonPanel, java.awt.BorderLayout.PAGE_END);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void CancelActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_CancelActionPerformed
        this.dispose();
    }//GEN-LAST:event_CancelActionPerformed

    private void CalculateActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_CalculateActionPerformed
        notifyMathListeners();
     
    }//GEN-LAST:event_CalculateActionPerformed

    /**
     * @param args the command line arguments
     */
//    public static void main(String args[]) {
//        /* Set the Nimbus look and feel */
//        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
//        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
//         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
//         */
//        try {
//            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
//                if ("Nimbus".equals(info.getName())) {
//                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
//                    break;
//                }
//            }
//        } catch (ClassNotFoundException ex) {
//            java.util.logging.Logger.getLogger(GatingHierarchy.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (InstantiationException ex) {
//            java.util.logging.Logger.getLogger(GatingHierarchy.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (IllegalAccessException ex) {
//            java.util.logging.Logger.getLogger(GatingHierarchy.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
//            java.util.logging.Logger.getLogger(GatingHierarchy.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
//        }
//        //</editor-fold>
//
//        /* Create and display the form */
//        java.awt.EventQueue.invokeLater(new Runnable() {
//            public void run() {
//                new GatingHierarchy().setVisible(true);
//            }
//        });
//    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JPanel ButtonPanel;
    private javax.swing.JButton Calculate;
    private javax.swing.JButton Cancel;
    private javax.swing.JTable GateMathTable;
    private javax.swing.JPanel classPanel;
    private javax.swing.JSpinner classSpinner;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JPanel jPanel3;
    private javax.swing.JPanel jPanel4;
    private javax.swing.JScrollPane jScrollPane;
    // End of variables declaration//GEN-END:variables

    private void updateTable(ArrayList<PolygonGate> gates) {

        RoiManager r = RoiManager.getInstance();
        int imageRoiCount = 0;

        if (!(r == null) && r.getCount() > 0) {
            imageRoiCount = r.getCount();
        }

        String[] availableGate = new String[gates.size() + imageRoiCount];

        for (int i = 0; i < gates.size(); i++) {

            PolygonGate gate = gates.get(i);
            availableGate[i] = "GATE-" + gate.getName();

        }

        if (!(r == null) && r.getCount() > 0) {
            Roi[] rois = r.getRoisAsArray();
            for (int i = gates.size(); i < r.getCount() + gates.size(); i++) {
                availableGate[i] = "ROI-" + rois[i - gates.size()].getName();
            }
        }

        GateMathTable = new JTable();
        GateMathTable.setModel(new javax.swing.table.DefaultTableModel(
                new Object[][]{
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null},
                    {null, null}
                },
                new String[]{
                    "Gate or ROI",
                    "Operation"

                }
        ) {
            Class[] types = new Class[]{
                java.lang.String.class, java.lang.String.class
            };
            boolean[] canEdit = new boolean[]{
                true, true
            };

            @Override
            public Class getColumnClass(int columnIndex) {
                return types[columnIndex];
            }

            @Override
            public boolean isCellEditable(int rowIndex, int columnIndex) {
                if (rowIndex < GateMathTable.getModel().getRowCount() - 1
                        && rowIndex > 0) {
                    if (columnIndex == 0
                            && !((GateMathTable.getModel().getValueAt(rowIndex - 1, 0)) == null)
                            && !((GateMathTable.getModel().getValueAt(rowIndex - 1, 1)) == null)) {
                        return true;
                    }
                    if (columnIndex == 1
                            && !((GateMathTable.getModel().getValueAt(rowIndex, 0)) == null)) {
                        return true;
                    }
                }
                if (rowIndex == 0 && columnIndex == 0) {
                    return true;
                }
                if (rowIndex == 0 && columnIndex == 1
                        && !((GateMathTable.getModel().getValueAt(rowIndex, 0)) == null)) {
                    return true;
                }

                if (columnIndex == 1 && GateMathTable.getModel().getRowCount() == rowIndex) {
                    return false;
                }
                return false;
            }
        });

        GateMathTable.getModel().addTableModelListener(this);

        GateMathTable.setAutoResizeMode(javax.swing.JTable.AUTO_RESIZE_OFF);
        GateMathTable.setMaximumSize(new java.awt.Dimension(370, 300));
        GateMathTable.setMinimumSize(new java.awt.Dimension(370, 300));
        GateMathTable.setPreferredSize(new java.awt.Dimension(370, 300));
        GateMathTable.setShowGrid(true);

        TableColumn column = null;
        column = GateMathTable.getColumnModel().getColumn(1);
        column.setCellEditor(new DefaultCellEditor(new JComboBox(GATEMATHOPTIONS)));
        column.setPreferredWidth(100);
        column = GateMathTable.getColumnModel().getColumn(0);
        column.setCellEditor(new DefaultCellEditor(new JComboBox(availableGate)));
        column.setPreferredWidth(270);

        GateMathTable.doLayout();
        GateMathTable.repaint();

        jScrollPane.setViewportView(GateMathTable);

    }

    @Override
    public void tableChanged(TableModelEvent e) {

        this.gateNames = new ArrayList<String>();
        this.opNames = new ArrayList<String>();

        if (!(GateMathTable.getModel().getValueAt(0, 0) == null)) {
            for (int row = 0; row < GateMathTable.getModel().getRowCount(); row++) {
                
                    if (!(GateMathTable.getModel().getValueAt(row, 0) == null) &&
                            !((String) GateMathTable.getModel().getValueAt(row, 0)).isEmpty()) {
                        gateNames.add((String) GateMathTable.getModel().getValueAt(row, 0));
                    }
                    if (!(GateMathTable.getModel().getValueAt(row, 1) == null) &&
                            !((String) GateMathTable.getModel().getValueAt(row, 1)).isEmpty()) {
                        opNames.add((String) GateMathTable.getModel().getValueAt(row, 1));
                    }
            }
        }
        if (gateNames.size()-opNames.size() == 1 && gateNames.size() > 1) {
            this.Calculate.setEnabled(true);
        } else {
            this.Calculate.setEnabled(false);
        }
    }


    
    public void addMathListener(AddClassByMathListener listener) {
        MathListeners.add(listener);
    }

    
    public void notifyMathListeners() {
        for (AddClassByMathListener listener : MathListeners) {
            System.out.println("PROFILING: starting math for class " + (int)classSpinner.getValue());
            listener.addClassByMath(gateNames, opNames, (int)classSpinner.getValue());
        }
    }
}
